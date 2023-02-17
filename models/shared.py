import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import gc
import nrrd
import torch
import torch.nn as nn
import torchio as tio

from multiprocessing import Pool
from numba import jit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset, DataLoader

class RawDataset:
    def __init__(self, data_dir, skip_norm=False):
        self.data_dir = data_dir

        if not skip_norm:
            intensity_transform = tio.transforms.RescaleIntensity()
            znorm_transform = tio.transforms.ZNormalization()

        data = []
        for file in os.listdir(data_dir):
            image, headers = nrrd.read(os.path.join(data_dir, file))

            tensor = torch.from_numpy(image).unsqueeze(0)
            if not skip_norm:
                tensor = intensity_transform(tensor)
                tensor = znorm_transform(tensor)

            data.append({
                "pid": os.path.splitext(file)[0].split("_")[0],
                "age": headers["age_days"],
                "tbv": headers["tbv"],
                "voxels": headers["tbv_n_voxels"],
                "voxel_volume": np.prod(headers["spacings"]),
                "raster": tensor
            })

        self.dataset = pd.DataFrame(columns=["pid", "age", "tbv", "voxels", "voxel_volume", "raster"], data=data)
        self.dataset[["age", "tbv", "voxels", "voxel_volume"]] = self.dataset[["age", "tbv", "voxels", "voxel_volume"]].apply(pd.to_numeric)

        self.age_minmax = MinMaxScaler()
        self.voxels_minmax = MinMaxScaler()
        self.age_std = StandardScaler()
        self.voxels_std = StandardScaler()

        self.dataset[["age"]] = self.age_minmax.fit_transform(self.dataset[["age"]])
        self.dataset[["voxels"]] = self.voxels_minmax.fit_transform(self.dataset[["voxels"]])
        self.dataset[["age"]] = self.age_std.fit_transform(self.dataset[["age"]])
        self.dataset[["voxels"]] = self.voxels_std.fit_transform(self.dataset[["voxels"]])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]

    def get_unique_pids(self):
        return self.dataset["pid"].unique()

    def get_indices_from_pids(self, pids):
        return self.dataset[self.dataset["pid"].isin(pids)].index
    
class RasterDataset(Dataset):
    def __init__(self, dataset, histogram_bins=None, drop_zero_bin=True):
        self.dataset = dataset
        self.histogram_bins = histogram_bins
        self.drop_zero_bin = drop_zero_bin

        if self.histogram_bins is not None:
            self.intensity_transform = tio.transforms.RescaleIntensity()
            self.znorm_transform = tio.transforms.ZNormalization()

            if self.drop_zero_bin:
                self.histogram_bins += 1
            
    def _get_histogram(self, raster):
        raster = raster.squeeze(0).numpy()

        kernels = np.lib.stride_tricks.sliding_window_view(np.pad(raster, 1), (3, 3, 3))
        hist = self._histogram_from_kernels(kernels, self.histogram_bins, self.drop_zero_bin)

        # Reshape to (bins, side_len, side_len, side_len)
        hist = torch.from_numpy(np.stack(np.split(hist, hist.shape[3], axis=3), axis=0).squeeze())
        
        # Normalize histogram
        hist = self.intensity_transform(hist)
        hist = self.znorm_transform(hist)

        return hist

    @staticmethod
    @jit(nopython=True) 
    def _histogram_from_kernels(kernels, bins, drop_zero_bin=True):
        # Using numba to speed up the triple loop

        hist = np.zeros((kernels.shape[0], kernels.shape[1], kernels.shape[2], bins))

        for i in range(kernels.shape[0]):
            for j in range(kernels.shape[1]):
                for k in range(kernels.shape[2]):
                    hist[i, j, k, :] = np.histogram(kernels[i, j, k], bins=bins, range=(0, 255))[0]
        
        return hist[:, :, :, 1:] if drop_zero_bin else hist

    def get_unique_pids(self):
        return self.raw_dataset.get_unique_pids()
    
    def get_indices_from_pids(self, pids):
        return self.raw_dataset.get_indices_from_pids(pids)

class TrainDataset(RasterDataset):
    def __init__(self, raw_dataset, pids, histogram_bins=None, drop_zero_bin=True):
        super().__init__(raw_dataset, histogram_bins, drop_zero_bin)

        self.pid_set = set(pids)
        self.volumes = self.dataset.get_indices_from_pids(pids)

        self.transform = tio.Compose([
            tio.transforms.RandomAffine(scales=0, degrees=10, translation=0.2, default_pad_value='minimum', p=0.25),
            tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
            tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.2),
            tio.transforms.RandomNoise(mean=0, std=0.1, p=0.2),
            tio.transforms.RandomBlur(std=0.1, p=0.2),
            tio.transforms.RandomBiasField(p=0.01),
            tio.transforms.RandomMotion(num_transforms=2, p=0.01),
            tio.transforms.RandomGhosting(num_ghosts=2, p=0.01),
            tio.transforms.RandomSwap(p=0.01),
            tio.transforms.RandomSpike(p=0.01)
        ])

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.dataset[self.volumes[idx]]
        raster = self.transform(entry["raster"])

        if self.histogram_bins is not None:
            raster = self._get_histogram(raster)

        return {
            "pid": entry["pid"], 
            "age": entry["age"], 
            "tbv": entry["tbv"], 
            "voxels": entry["voxels"], 
            "voxel_volume": entry["voxel_volume"], 
            "raster": raster
        }

    
class ValDataset(RasterDataset):
    def __init__(self, raw_dataset, pids, histogram_bins=None, drop_zero_bin=True):
        super().__init__(raw_dataset, histogram_bins, drop_zero_bin)

        self.pid_set = set(pids)
        self.volumes = self.dataset.get_indices_from_pids(pids)
        
        # Rasters can be preprocessed bc they won't be transformed
        # TODO: Add multiprocessing not to wait 15 seconds...
        self.preprocessed_rasters = {}
        for idx in self.volumes:
            self._preprocess_raster(idx)

    def _preprocess_raster(self, idx):
        entry = self.dataset[idx]
        raster = entry["raster"]

        if self.histogram_bins is not None:
            raster = self._get_histogram(raster)

        self.preprocessed_rasters[idx] = raster

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.volumes[idx]
        entry = self.dataset[idx]
        raster = self.preprocessed_rasters[idx]

        return {
            "pid": entry["pid"], 
            "age": entry["age"], 
            "tbv": entry["tbv"], 
            "voxels": entry["voxels"], 
            "voxel_volume": entry["voxel_volume"], 
            "raster": raster
        }

class EarlyStopper:
    def __init__(self, patience=25, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if model: self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model: self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_tbv_age(model, criterion, optimizer, tr_dataloader, val_dataloader, early_stopper, 
                  num_epochs, device, dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0), 
                  grad_clip=5, trace_func=print, scheduler=None):
    
    train_loss_list = pd.Series(dtype=np.float32)
    val_loss_list = pd.Series(dtype=np.float32)

    has_do = hasattr(model, "do")
    dropout_value = model.do.p if has_do else 0.0

    for epoch in range(num_epochs):
        if has_do and epoch % dropout_change_epochs == 0 and epoch != 0:
            dropout_value += dropout_change
            dropout_value = max(dropout_value, dropout_range[0])
            dropout_value = min(dropout_value, dropout_range[1])
            model.do.p = dropout_value
            
        training_loss = 0.
        
        model.train()
        for i, data in enumerate(tr_dataloader):
            rasters = data["raster"].float().to(device)
            voxels = data["voxels"].float().to(device)
            ages = data["age"].float().to(device)

            ground = torch.dstack((voxels, ages)).squeeze()

            optimizer.zero_grad()

            # split the predictions into predicted voxels and ages
            predictions = model(rasters).squeeze()
            loss = criterion(predictions, ground)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Avoid exploding gradients
            optimizer.step()

            training_loss += loss.item()

            torch.cuda.empty_cache()
            gc.collect()

            if (i+1) % 5 == 0:
                trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Step [{i+1}/{len(tr_dataloader)}], Loss: {loss.item():.4f}                                                            ", end="\r")

        validation_loss = 0.
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                rasters = data["raster"].float().to(device)
                voxels = data["voxels"].float().to(device)
                ages = data["age"].float().to(device)

                ground = torch.dstack((voxels, ages)).squeeze()
                predictions = model(rasters).squeeze()
                loss = criterion(predictions, ground)

                validation_loss += loss.item()

        train_loss = training_loss/len(tr_dataloader)
        val_loss = validation_loss/len(val_dataloader)
        train_loss_list.at[epoch] = train_loss
        val_loss_list.at[epoch] = val_loss

        if(scheduler):  scheduler.step(train_loss)

        stopping = early_stopper(validation_loss, model)
        trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Dropout p: {dropout_value:.2f}, Patience: {early_stopper.counter}/{early_stopper.patience}            ")
        if stopping:
            break
    
    return train_loss_list, val_loss_list

def train_tbv(model, criterion, optimizer, tr_dataloader, val_dataloader, early_stopper, 
              num_epochs, device, dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0), 
              grad_clip=5, trace_func=print, scheduler=None):

    train_loss_list = pd.Series(dtype=np.float32)
    val_loss_list = pd.Series(dtype=np.float32)

    has_do = hasattr(model, "do")
    dropout_value = model.do.p if has_do else 0.0

    for epoch in range(num_epochs):
        if has_do and epoch % dropout_change_epochs == 0 and epoch != 0:
            dropout_value += dropout_change
            dropout_value = max(dropout_value, dropout_range[0])
            dropout_value = min(dropout_value, dropout_range[1])
            model.do.p = dropout_value
            
        training_loss = 0.
        
        model.train()
        for i, data in enumerate(tr_dataloader):
            rasters = data["raster"].float().to(device)
            voxels = data["voxels"].float().to(device)

            optimizer.zero_grad()

            predictions = model(rasters).squeeze()
            loss = criterion(predictions, voxels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Avoid exploding gradients
            optimizer.step()

            training_loss += loss.item()

            torch.cuda.empty_cache()
            gc.collect()

            if (i+1) % 5 == 0:
                trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Step [{i+1}/{len(tr_dataloader)}], Loss: {loss.item():.4f}                                                            ", end="\r")
            
        validation_loss = 0.
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                rasters = data["raster"].float().to(device)
                voxels = data["voxels"].float().to(device)
                
                predictions = model(rasters).squeeze()
                loss = criterion(predictions, voxels)

                validation_loss += loss.item()

        train_loss = training_loss/len(tr_dataloader)
        val_loss = validation_loss/len(val_dataloader)
        train_loss_list.at[epoch] = train_loss
        val_loss_list.at[epoch] = val_loss

        if(scheduler):  scheduler.step(train_loss)

        stopping = early_stopper(validation_loss, model)
        trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Dropout p: {dropout_value:.2f}, Patience: {early_stopper.counter}/{early_stopper.patience}            ")
        if stopping:
            break
    
    return train_loss_list, val_loss_list

def final_eval_tbv(model, dataloader, raw_dataset, device, trace_func=print, verbose=False):
    all_diffs = []
    model.eval()

    if(verbose):
        trace_func("\nEvaluating model on test set...")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            rasters = data["raster"].float().to(device)
            tbvs = data["tbv"].float().numpy().reshape(-1, 1)

            predictions = model(rasters).squeeze().cpu()
            predictions = predictions.detach().numpy().reshape(-1, 1)
            
            # Revert the normalization and standardization
            predictions = raw_dataset.voxels_std.inverse_transform(predictions)
            predictions = raw_dataset.voxels_minmax.inverse_transform(predictions)
            predicted_tbvs = predictions * data["voxel_volume"].numpy().reshape(-1, 1)

            tbv_diffs = np.abs(predicted_tbvs - tbvs)
            all_diffs.append(tbv_diffs)

            if(verbose and i < 5):
                for j in range(len(tbvs)):
                    trace_func(f"Predicted TBV: {predicted_tbvs[j][0]:.2f}\tActual TBV: {tbvs[j][0]:.2f}\tDifference: {tbv_diffs[j][0]:.2f}")

    all_diffs = np.concatenate(all_diffs)
    
    avg_diff = round(np.mean(all_diffs), 2)
    std_diff = round(np.std(all_diffs), 2)

    if(verbose):
        trace_func()
        trace_func(f"Average difference: {avg_diff} cc.\tStandard deviation: {std_diff} cc.")

    return avg_diff, std_diff

def final_eval_tbv_age(model, dataloader, raw_dataset, device, trace_func=print, verbose=False):
    all_diffs_tbv = []
    all_diffs_age = []
    model.eval()

    if(verbose):
        trace_func("\nEvaluating model on test set...")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            rasters = data["raster"].float().to(device)
            tbvs = data["tbv"].float().numpy().reshape(-1, 1)
            ages = data["age"].float().numpy().reshape(-1, 1)

            predictions = model(rasters).squeeze().cpu()
            pred_voxels, pred_ages = torch.split(predictions, 1, dim=1)
            pred_voxels = pred_voxels.detach().numpy().reshape(-1, 1)
            pred_ages = pred_ages.detach().numpy().reshape(-1, 1)
            
            # Revert the normalization and standardization
            pred_voxels = raw_dataset.voxels_std.inverse_transform(pred_voxels)
            pred_voxels = raw_dataset.voxels_minmax.inverse_transform(pred_voxels)
            pred_ages = raw_dataset.age_std.inverse_transform(pred_ages)
            pred_ages = raw_dataset.age_minmax.inverse_transform(pred_ages)
            ages = raw_dataset.age_std.inverse_transform(ages)
            ages = raw_dataset.age_minmax.inverse_transform(ages)
            
            pred_tbvs = pred_voxels * data["voxel_volume"].numpy().reshape(-1, 1)

            tbv_diffs = np.abs(pred_tbvs - tbvs)
            age_diffs = np.abs(pred_ages - ages)

            all_diffs_tbv.append(tbv_diffs)
            all_diffs_age.append(age_diffs)

            if(verbose and i < 5):
                for j in range(len(tbvs)):
                    trace_func(f"Predicted TBV: {pred_tbvs[j][0]:.2f}\tActual TBV: {tbvs[j][0]:.2f}\tDifference: {tbv_diffs[j][0]:.2f}")
                    trace_func(f"Predicted Age: {pred_ages[j][0]:.2f}\tActual Age: {ages[j][0]:.2f}\tDifference: {age_diffs[j][0]:.2f}")

    all_diffs_tbv = np.concatenate(all_diffs_tbv)
    all_diffs_age = np.concatenate(all_diffs_age)
    
    avg_diff_tbv = round(np.mean(all_diffs_tbv), 2)
    std_diff_tbv = round(np.std(all_diffs_tbv), 2)
    avg_diff_age = round(np.mean(all_diffs_age), 2)
    std_diff_age = round(np.std(all_diffs_age), 2)

    if(verbose):
        trace_func()
        trace_func(f"Average difference TBV: {avg_diff_tbv} cc.\tStandard deviation TBV: {std_diff_tbv} cc.")
        trace_func(f"Average difference Age: {avg_diff_age} days.\tStandard deviation Age: {std_diff_age} days.")

    return avg_diff_tbv, std_diff_tbv, avg_diff_age, std_diff_age

def run(model, raw_dataset, device, optimizer_class, criterion_class, train_fun, final_eval_fun, 
        optimizer_params={}, criterion_params={}, num_epochs=400, patience=30, batch_size=8, data_workers=4, 
        trace_func=print, scheduler_class=None, scheduler_params={}, k_fold=6, histogram_bins=None, drop_zero_bin=False, 
        dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0), grad_clip=5, override_val_pids=None):

    # Save model to reset it after each fold
    torch.save(model.state_dict(), "base_weights.pt")

    train_score = pd.Series(dtype=np.float32)
    val_score = pd.Series(dtype=np.float32)
    unscaled_loss = pd.Series(dtype=np.float32)
    
    unique_pids = raw_dataset.get_unique_pids()
    total_pids = len(unique_pids)
    fraction = 1/k_fold
    seg = int(total_pids*fraction)
    
    # tr:train, val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trace_func(f"Fold {i+1}/{k_fold}")

        model.load_state_dict(torch.load("base_weights.pt"))

        criterion = criterion_class(**criterion_params)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = None if not scheduler_class else scheduler_class(optimizer, **scheduler_params)
        early_stopper = EarlyStopper(patience=patience, verbose=False, path="best_weights.pt", trace_func=trace_func)

        if override_val_pids:
            val_pids = override_val_pids
            train_pids = [pid for pid in unique_pids if pid not in val_pids]
        else:
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_pids
            
            train_left_indices = list(range(trll,trlr))
            train_right_indices = list(range(trrl,trrr))
            
            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall,valr))

            val_pids = unique_pids[val_indices]
            train_pids = unique_pids[train_indices]
        
        train_set = TrainDataset(raw_dataset, train_pids, histogram_bins=histogram_bins, drop_zero_bin=drop_zero_bin)
        val_set = ValDataset(raw_dataset, val_pids, histogram_bins=histogram_bins, drop_zero_bin=drop_zero_bin)

        trace_func(f"Train volumes: {len(train_set)}")
        trace_func(f"Validation volumes: {len(val_set)}")

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=data_workers, drop_last=True, pin_memory=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=data_workers, drop_last=True, pin_memory=True)
        
        train_loss_list, val_loss_list = train_fun(
            model, criterion, optimizer, train_dataloader, val_dataloader, 
            early_stopper, num_epochs, device, trace_func=trace_func, scheduler=scheduler,
            dropout_change=dropout_change, dropout_change_epochs=dropout_change_epochs, dropout_range=dropout_range, 
            grad_clip=grad_clip
        )
        
        model.load_state_dict(torch.load("best_weights.pt"))

        train_score.at[i] = train_loss_list
        val_score.at[i] = val_loss_list
        unscaled_loss.at[i] = final_eval_fun(model, val_dataloader, raw_dataset, device, trace_func=trace_func, verbose=True)

        if override_val_pids:
            # If we are overriding the validation set, we don't want to do any more folds
            break
    
    os.remove("base_weights.pt")
    
    return train_score, val_score, unscaled_loss