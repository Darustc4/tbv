import pandas as pd
import numpy as np

import os
import gc
import scipy
import nrrd
import torch
import torchio as tio

from enum import Enum
from numba import jit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset, DataLoader


class RawDataset:
    def __init__(self, data_dir, side_len=96, crop_factor=0.8, verbose=True):
        self.data_dir = data_dir
        self.side_len = np.array([side_len])*3
        self.crop_factor = crop_factor

        intensity_transform = tio.transforms.RescaleIntensity()
        znorm_transform = tio.transforms.ZNormalization()

        data = []
        for file in os.listdir(data_dir):
            raster, headers = nrrd.read(os.path.join(data_dir, file))
            tensor = torch.from_numpy(raster).unsqueeze(0)
            tensor = intensity_transform(tensor)
            tensor = znorm_transform(tensor)

            voxel_vol = np.prod(headers["spacings"])
            data.append({
                "pid": os.path.splitext(file)[0].split("_")[0],
                "age": headers["age_days"],
                "tbv": headers["tbv"],
                "avg_voxels": self._get_avg_voxels(raster, headers["tbv"], voxel_vol),
                "voxel_vol": voxel_vol,
                "raster": tensor
            })
        
        # Note: The voxels the brain occupies are calculated on the fly depending on cropping and zooming 
        #       'avg_voxels' is only meant for data standardization, not for prediction

        self.dataset = pd.DataFrame(columns=["pid", "age", "tbv", "avg_voxels", "voxel_vol", "raster"], data=data)
        self.dataset[["age", "tbv", "avg_voxels", "voxel_vol"]] = self.dataset[["age", "tbv", "avg_voxels", "voxel_vol"]].apply(pd.to_numeric)

        self.age_minmax = MinMaxScaler()
        self.voxels_minmax = MinMaxScaler()
        self.age_std = StandardScaler()
        self.voxels_std = StandardScaler()

        self.dataset[["age"]] = self.age_minmax.fit_transform(self.dataset[["age"]])
        self.dataset[["voxels"]] = self.voxels_minmax.fit_transform(self.dataset[["voxels"]])
        self.dataset[["age"]] = self.age_std.fit_transform(self.dataset[["age"]])
        self.dataset[["voxels"]] = self.voxels_std.fit_transform(self.dataset[["voxels"]])
        
        if verbose:
            self.print_stats()
        
        if self.compute_prev_scans:
            prev_scans = []
            for _, scan in self.dataset.iterrows():
                prev_scans.append(self._get_previous_scans(scan["pid"], scan["age"], add_noise=False))
            self.dataset["prev_scans"] = prev_scans
    
    def _get_avg_voxels(self, raster, tbv, voxel_vol):
        original_voxels = tbv / voxel_vol

        cropped_side_len = self._crop_raster(raster, return_shape_only=True, force_crop_factor=self.crop_factor)[0]
        resize_factor = self.side_len / cropped_side_len
        cropped_voxel_vol = voxel_vol / resize_factor
        
        cropped_voxels = tbv / cropped_voxel_vol

        return (original_voxels + cropped_voxels) / 2
        
    
    def _crop_raster(self, raster, return_shape_only=False, force_crop_factor=None):
        if force_crop_factor is None:
            crop_shape = (raster.shape * np.random.uniform(self.crop_factor, 1.0, size=3) * (1, np.random.uniform(self.crop_factor, 1.0), 1)).astype(int)
        else:
            crop_shape = (raster.shape * (force_crop_factor, force_crop_factor**2, force_crop_factor)).astype(int)
            
        max_value = np.max(crop_shape.shape)
        pad_shape = (max_value, max_value, max_value)

        if return_shape_only: return pad_shape
        
        crop = tio.CropOrPad(target_shape=crop_shape)
        pad = tio.CropOrPad(target_shape=pad_shape)
        
        return pad(crop(np.expand_dims(raster, axis=0))).squeeze()
    
    def _prepare_raster(self, raster, voxel_spacings, tbv):
        old_sizes = np.array(raster.shape)
        old_spacings = voxel_spacings
        new_sizes = self.side_len
        
        total_volume = np.prod(old_sizes) * np.prod(old_spacings)
        
        if np.array_equal(old_sizes, new_sizes):
            return raster, voxel_spacings, round(tbv * np.prod(new_sizes) / total_volume, 2)
        
        resize_factors = new_sizes / old_sizes

        raster = scipy.ndimage.zoom(raster, resize_factors)
        new_spacings = old_spacings / resize_factors

        return raster, new_spacings
    
    def _get_previous_scans(self, pid, age, add_noise=False):
        prev_scans = self.get_previous_scan_metadata(pid, age, self.prev_scans_n)
        n_features = self.prev_scans_n if self.prev_scans_no_age else self.prev_scans_n*2

        if prev_scans is None:
            return np.zeros((n_features))
        
        if self.prev_scans_no_age:  prev_scans = prev_scans[['voxels']].to_numpy().flatten()
        else:                       prev_scans = prev_scans[['voxels', 'age']].to_numpy().flatten()
        
        if add_noise:
            prev_scans[['voxels']] += np.random.normal(0, 0.05, prev_scans[['voxels']].shape)

        prev_scans = np.pad(prev_scans, (0, n_features - len(prev_scans)), 'constant', constant_values=0)
        
        return prev_scans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]

    def print_stats(self):
        print("Label standardization parameters:")
        print(f"Voxels min: {self.voxels_minmax.data_min_}, max: {self.voxels_minmax.data_max_}")
        print(f"Voxels mean: {self.voxels_std.mean_}, std: {self.voxels_std.scale_}")
        print()
        print(f"Age min: {self.age_minmax.data_min_}, max: {self.age_minmax.data_max_}")
        print(f"Age mean: {self.age_std.mean_}, std: {self.age_std.scale_}")
        print()

    def get_unique_pids(self):
        return self.dataset["pid"].unique()

    def get_indices_from_pids(self, pids):
        return self.dataset[self.dataset["pid"].isin(pids)].index
    
    def get_previous_scan_metadata(self, pid, age, n=3):
        previous_scans = self.dataset[(self.dataset["pid"] == pid) & (self.dataset["age"] < age)].sort_values("age", ascending=False)
        if self.prev_scans_age_as_diff: previous_scans["age"] = previous_scans["age"] - age
        return previous_scans.iloc[:n].drop(columns=["raster"]) if len(previous_scans) > 0 else None
    
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
            "raster": raster,
            "prev_scans": entry["prev_scans"] if "prev_scans" in entry else None
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
            "raster": raster,
            "prev_scans": entry["prev_scans"] if "prev_scans" in entry else None
        }

class EarlyStopper:
    def __init__(self, patience=25, verbose=False, delta=0, ignore_first_epochs=0, path='checkpoint.pt', trace_func=print):
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
        self.ignore_first_epochs = ignore_first_epochs
        self.path = path
        self.trace_func = trace_func

    def __call__(self, epoch, val_loss, model=None):
        if epoch < self.ignore_first_epochs:
            return False

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

class TrainMode(Enum):
    NO_AGE = 0
    IN_AGE = 1
    OUT_AGE = 2
    PREV_SCANS = 3

def train(model, criterion, optimizer, tr_dataloader, val_dataloader, early_stopper, train_mode, 
                  num_epochs, device, dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0), 
                  grad_clip=5, trace_func=print, scheduler=None):
    
    train_loss_list = pd.Series(dtype=np.float32)
    val_loss_list = pd.Series(dtype=np.float32)

    has_do = hasattr(model, "do")
    dropout_value = model.do.p if has_do else 0.0

    for epoch in range(num_epochs):
        if has_do and epoch % dropout_change_epochs == 0:
            if epoch != 0: dropout_value += dropout_change
            dropout_value = max(dropout_value, dropout_range[0])
            dropout_value = min(dropout_value, dropout_range[1])
            model.do.p = dropout_value
            
        training_loss = 0.
        
        model.train()
        for i, data in enumerate(tr_dataloader):
            rasters = data["raster"].float().to(device)
            voxels = data["voxels"].float().to(device)
            
            if train_mode != TrainMode.NO_AGE:
                ages = data["age"].float().to(device)
            
            if train_mode == TrainMode.PREV_SCANS:
                prev_scans = data["prev_scans"].float().to(device)
                
            optimizer.zero_grad()
            
            if train_mode == TrainMode.NO_AGE:
                predictions = model(rasters).squeeze()
                loss = criterion(predictions, voxels)
            elif train_mode == TrainMode.IN_AGE:
                predictions = model(rasters, ages.unsqueeze(1)).squeeze()
                loss = criterion(predictions, voxels)
            elif train_mode == TrainMode.OUT_AGE:
                ground = torch.dstack((voxels, ages)).squeeze()
                predictions = model(rasters).squeeze()
                loss = criterion(predictions, ground)
            elif train_mode == TrainMode.PREV_SCANS:
                predictions = model(rasters, ages.unsqueeze(1), prev_scans).squeeze()
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
                
                if train_mode != TrainMode.NO_AGE:
                    ages = data["age"].float().to(device)
                
                if train_mode == TrainMode.PREV_SCANS:
                    prev_scans = data["prev_scans"].float().to(device)

                if train_mode == TrainMode.NO_AGE:
                    predictions = model(rasters).squeeze()
                    loss = criterion(predictions, voxels)
                elif train_mode == TrainMode.IN_AGE:
                    predictions = model(rasters, ages.reshape(-1, 1)).squeeze()
                    loss = criterion(predictions, voxels)
                elif train_mode == TrainMode.OUT_AGE:
                    ground = torch.dstack((voxels, ages)).squeeze()
                    predictions = model(rasters).squeeze()
                    loss = criterion(predictions, ground)
                elif train_mode == TrainMode.PREV_SCANS:
                    predictions = model(rasters, ages.unsqueeze(1), prev_scans).squeeze()
                    loss = criterion(predictions, voxels)

                validation_loss += loss.item()

        train_loss = training_loss/len(tr_dataloader)
        val_loss = validation_loss/len(val_dataloader)
        train_loss_list.at[epoch] = train_loss
        val_loss_list.at[epoch] = val_loss

        if(scheduler):  scheduler.step(train_loss)

        stopping = early_stopper(epoch, val_loss, model)
        trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Dropout p: {dropout_value:.2f}, Patience: {early_stopper.counter}/{early_stopper.patience}            ")
        if stopping:
            break
    
    return train_loss_list, val_loss_list

def final_eval(model, dataloader, raw_dataset, device, train_mode, trace_func=print, verbose=False):
    all_diffs_tbv = []
    all_diffs_age = []
    model.eval()

    if(verbose):
        trace_func("\nEvaluating model on test set...")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            rasters = data["raster"].float().to(device)
            tbvs = data["tbv"].float().numpy().reshape(-1, 1)
            
            if train_mode != TrainMode.NO_AGE:
                ages = data["age"].float().reshape(-1, 1)

            if train_mode == TrainMode.PREV_SCANS:
                prev_scans = data["prev_scans"].float().to(device)
            
            if train_mode == TrainMode.NO_AGE:
                pred_voxels = model(rasters).squeeze().cpu()
            elif train_mode == TrainMode.IN_AGE:
                pred_voxels = model(rasters, ages.to(device)).squeeze().cpu()
                ages = ages.cpu()
            elif train_mode == TrainMode.OUT_AGE:
                predictions = model(rasters).squeeze().cpu()
                pred_voxels, pred_ages = torch.split(predictions, 1, dim=1)
                pred_ages = pred_ages.detach().numpy().reshape(-1, 1)
            elif train_mode == TrainMode.PREV_SCANS:
                pred_voxels = model(rasters, ages.to(device), prev_scans).squeeze().cpu()
                ages = ages.cpu()
            
            pred_voxels = pred_voxels.detach().numpy().reshape(-1, 1)
            
            # Revert the normalization and standardization
            pred_voxels = raw_dataset.voxels_std.inverse_transform(pred_voxels)
            pred_voxels = raw_dataset.voxels_minmax.inverse_transform(pred_voxels)
            
            if train_mode == TrainMode.OUT_AGE:
                pred_ages = raw_dataset.age_std.inverse_transform(pred_ages)
                pred_ages = raw_dataset.age_minmax.inverse_transform(pred_ages)
                ages = raw_dataset.age_std.inverse_transform(ages)
                ages = raw_dataset.age_minmax.inverse_transform(ages)
                
            pred_tbvs = pred_voxels * data["voxel_volume"].numpy().reshape(-1, 1)

            tbv_diffs = np.abs(pred_tbvs - tbvs)
            all_diffs_tbv.append(tbv_diffs)
            
            if train_mode == TrainMode.OUT_AGE:
                age_diffs = np.abs(pred_ages - ages)
                all_diffs_age.append(age_diffs)

            if(verbose and i < 5):
                for j in range(len(tbvs)):
                    trace_func(f"Predicted TBV: {pred_tbvs[j][0]:.2f}\tActual TBV: {tbvs[j][0]:.2f}\tDifference: {tbv_diffs[j][0]:.2f}")
                    if train_mode == TrainMode.OUT_AGE:
                        trace_func(f"Predicted Age: {pred_ages[j][0]:.2f}\tActual Age: {ages[j][0]:.2f}\tDifference: {age_diffs[j][0]:.2f}")

    all_diffs_tbv = np.concatenate(all_diffs_tbv)
    avg_diff_tbv = round(np.mean(all_diffs_tbv), 2)
    std_diff_tbv = round(np.std(all_diffs_tbv), 2)
    
    if train_mode == TrainMode.OUT_AGE:
        all_diffs_age = np.concatenate(all_diffs_age)
        avg_diff_age = round(np.mean(all_diffs_age), 2)
        std_diff_age = round(np.std(all_diffs_age), 2)

    if(verbose):
        trace_func()
        trace_func(f"Average difference TBV: {avg_diff_tbv} cc.\tStandard deviation TBV: {std_diff_tbv} cc.")
        if train_mode == TrainMode.OUT_AGE:
            trace_func(f"Average difference Age: {avg_diff_age} days.\tStandard deviation Age: {std_diff_age} days.")

    if train_mode == TrainMode.OUT_AGE:
        return avg_diff_tbv, std_diff_tbv, avg_diff_age, std_diff_age
    else:
        return avg_diff_tbv, std_diff_tbv

def run(model, raw_dataset, device, optimizer_class, criterion_class, train_mode, 
        optimizer_params={}, criterion_params={}, num_epochs=500, patience=100, early_stop_ignore_first_epochs= 100, 
        batch_size=8, data_workers=4, trace_func=print, scheduler_class=None, scheduler_params={}, k_fold=6, 
        histogram_bins=None, drop_zero_bin=False, grad_clip=5, override_val_pids=None,
        dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0)):

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
        early_stopper = EarlyStopper(
            patience=patience, ignore_first_epochs=early_stop_ignore_first_epochs, 
            verbose=False, path="best_weights.pt", trace_func=trace_func
        )

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

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=data_workers, pin_memory=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=data_workers, pin_memory=True)
        
        train_loss_list, val_loss_list = train(
            model, criterion, optimizer, train_dataloader, val_dataloader, early_stopper, 
            train_mode, num_epochs, device, trace_func=trace_func, scheduler=scheduler,
            dropout_change=dropout_change, dropout_change_epochs=dropout_change_epochs, dropout_range=dropout_range, 
            grad_clip=grad_clip
        )
        
        model.load_state_dict(torch.load("best_weights.pt"))

        train_score.at[i] = train_loss_list
        val_score.at[i] = val_loss_list
        unscaled_loss.at[i] = final_eval(model, val_dataloader, raw_dataset, device, train_mode, trace_func=trace_func, verbose=True)

        if override_val_pids:
            # If we are overriding the validation set, we don't want to do any more folds
            break
    
    os.remove("base_weights.pt")
    
    return train_score, val_score, unscaled_loss