import pandas as pd
import numpy as np

import os
import gc
import scipy
import nrrd
import torch
import torchio as tio

from enum import Enum

from torch.utils.data import Dataset, DataLoader


class RawDataset:
    def __init__(self, data_dir, side_len=96, crop_factor=0.8, crop_chance=0.3, verbose=True):
        self.data_dir = data_dir
        self.side_len = side_len
        self.crop_factor = crop_factor
        self.crop_chance = crop_chance

        intensity_transform = tio.transforms.RescaleIntensity()
        znorm_transform = tio.transforms.ZNormalization()

        self.voxels_min = np.inf
        self.voxels_max = -np.inf
        self.voxels_mean = 0
        self.voxels_std = 0

        self.age_min = np.inf
        self.age_max = -np.inf
        self.age_mean = 0
        self.age_std = 0

        data = []
        for file in os.listdir(data_dir):
            raster, headers = nrrd.read(os.path.join(data_dir, file))
            tensor = torch.from_numpy(raster).unsqueeze(0)
            tensor = intensity_transform(tensor)
            tensor = znorm_transform(tensor)

            pid = os.path.splitext(file)[0].split("_")[0]
            voxel_vol = np.prod(headers["spacings"])
            tbv = float(headers["tbv"])
            age = int(headers["age_days"])

            upper_voxels, lower_voxels = self._get_voxel_range(tensor, tbv, voxel_vol)
            data.append({
                "pid": pid,
                "age": age,
                "tbv": tbv,
                "avg_voxels": (upper_voxels + lower_voxels) / 2,
                "voxel_vol": voxel_vol,
                "raster": tensor
            })

            if verbose:
                print(f"Loaded {pid} with {lower_voxels:.2f} - {upper_voxels:.2f} voxels. Age: {age}, TBV: {tbv}, Voxel Volume: {voxel_vol:.3f} cm3.")

            self.voxels_min = min(self.voxels_min, lower_voxels)
            self.voxels_max = max(self.voxels_max, upper_voxels)
            self.age_min = min(self.age_min, age)
            self.age_max = max(self.age_max, age)
        
        # Note: The voxels the brain occupies are calculated on the fly depending on cropping and zooming 
        #       'avg_voxels' is only meant for data standardization, not for prediction. 
        #       It is dropped after we have the mean and std for the dataset. 

        self.dataset = pd.DataFrame(columns=["pid", "age", "tbv", "avg_voxels", "voxel_vol", "raster"], data=data)
        self.dataset[["age", "tbv", "avg_voxels", "voxel_vol"]] = self.dataset[["age", "tbv", "avg_voxels", "voxel_vol"]].apply(pd.to_numeric)

        self.voxels_mean = self.dataset["avg_voxels"].mean()
        self.voxels_std = self.dataset["avg_voxels"].std()
        self.age_mean = self.dataset["age"].mean()
        self.age_std = self.dataset["age"].std()

        self.voxels_mean = (self.voxels_mean - self.voxels_min) / (self.voxels_max - self.voxels_min)
        self.voxels_std = self.voxels_std / (self.voxels_max - self.voxels_min)
        self.age_mean = (self.age_mean - self.age_min) / (self.age_max - self.age_min)
        self.age_std = self.age_std / (self.age_max - self.age_min)

        self.dataset["age"] = self.normalize_age(self.dataset["age"])

        self.dataset.drop(columns=["avg_voxels"], inplace=True)

        if verbose:
            self.print_stats()
    
    def _get_voxel_range(self, raster, tbv, voxel_vol):
        original_voxels = tbv / voxel_vol

        cropped_side_len = self._reshape_raster(raster, return_shape_only=True, force_crop_factor=self.crop_factor)[0]
        resize_factor = self.side_len / cropped_side_len
        cropped_voxel_vol = voxel_vol / resize_factor
        
        cropped_voxels = tbv / cropped_voxel_vol

        return original_voxels, cropped_voxels
        
    def _reshape_raster(self, raster, return_shape_only=False, force_crop_factor=None, no_crop=False):
        # Crop the raster randomly and then pad it to be a cube. No deformation.

        if no_crop:
            max_value = np.max(raster.shape)
        else:
            if force_crop_factor is None:
                crop_factor = np.random.uniform(self.crop_factor, 1.0, size=3)
                #crop_factor[2] *= np.random.uniform(self.crop_factor, 1.0) # The y axis usually has more unused space. Crop further.
                crop_shape = (raster.shape[1:] * crop_factor).astype(int)
            else:
                crop_shape = (raster.shape[1:] * np.array([force_crop_factor, force_crop_factor**2, force_crop_factor])).astype(int)        
            max_value = np.max(crop_shape)
        
        pad_shape = (max_value, max_value, max_value)
        if return_shape_only: return pad_shape
        
        if no_crop: 
            pad = tio.CropOrPad(target_shape=pad_shape)
            return pad(raster)
        else:
            crop = tio.CropOrPad(target_shape=crop_shape)
            pad = tio.CropOrPad(target_shape=pad_shape)
    
            return pad(crop(raster))
        
    def _prep_raster(self, raster, voxel_vol, new_side_len=100, no_crop=False):
        original_raster = self._reshape_raster(raster, no_crop=no_crop)
        original_side_len = original_raster.shape[-1]
        original_voxel_vol = voxel_vol

        if np.array_equal(original_side_len, new_side_len):
            return original_raster, original_voxel_vol
        
        # Resize the raster to be a cube of side length new_side_len
        resize_factor = new_side_len / original_side_len

        new_raster = torch.from_numpy(scipy.ndimage.zoom(original_raster.squeeze(0), (resize_factor, resize_factor, resize_factor), order=1)).unsqueeze(0)
        new_voxel_vol = original_voxel_vol / resize_factor

        return new_raster, new_voxel_vol

    def __len__(self):
        return len(self.dataset)

    def get(self, idx, no_crop=False):
        no_crop = no_crop or (np.random.uniform() > self.crop_chance)

        item = self.dataset.iloc[idx]
        raster, voxel_vol = self._prep_raster(item["raster"], item["voxel_vol"], new_side_len=self.side_len, no_crop=no_crop)
        voxel_count = self.normalize_voxels(item["tbv"] / voxel_vol)

        item = {
            "age": item["age"],
            "tbv": item["tbv"],
            "voxel_vol": voxel_vol,
            "voxels": voxel_count,
            "raster": raster
        }

        return item

    def print_stats(self):
        print("Label standardization parameters:")
        print(f"Voxels min: {self.voxels_min:.2f}, max: {self.voxels_max:.2f}")
        print(f"Voxels mean: {self.voxels_mean:.2f}, std: {self.voxels_std:.2f}")
        print()
        print(f"Age min: {self.age_min:.2f}, max: {self.age_max:.2f}")
        print(f"Age mean: {self.age_mean:.2f}, std: {self.age_std:.2f}")
        print()

    def get_unique_pids(self):
        return self.dataset["pid"].unique()

    def get_indices_from_pids(self, pids):
        return self.dataset[self.dataset["pid"].isin(pids)].index

    def normalize_voxels(self, voxels, inverse=False):
        if inverse:
            voxels = voxels * self.voxels_std + self.voxels_mean
            voxels = voxels * (self.voxels_max - self.voxels_min) + self.voxels_min
        else:
            voxels = (voxels - self.voxels_min) / (self.voxels_max - self.voxels_min)
            voxels = (voxels - self.voxels_mean) / self.voxels_std
        
        return voxels

    def normalize_age(self, age, inverse=False):
        if inverse:
            age = age * self.age_std + self.age_mean
            age = age * (self.age_max - self.age_min) + self.age_min
        else:
            age = (age - self.age_min) / (self.age_max - self.age_min)
            age = (age - self.age_mean) / self.age_std
        
        return age
    
class RasterDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def get_unique_pids(self):
        return self.raw_dataset.get_unique_pids()
    
    def get_indices_from_pids(self, pids):
        return self.raw_dataset.get_indices_from_pids(pids)

    def normalize_voxels(self, voxels, inverse=False):
        return self.raw_dataset.normalize_voxels(voxels, inverse)
    
    def normalize_age(self, age, inverse=False):
        return self.raw_dataset.normalize_age(age, inverse)
    
class TrainDataset(RasterDataset):
    def __init__(self, raw_dataset, pids, verbose=True):
        super().__init__(raw_dataset)

        self.pid_set = set(pids)
        self.volumes = self.raw_dataset.get_indices_from_pids(pids)

        self.transform = tio.Compose([
            tio.OneOf({
                tio.transforms.RandomAffine(scales=0, degrees=15, translation=0, default_pad_value='minimum'): 0.8,
                tio.transforms.RandomAffine(scales=0, degrees=90, translation=0, default_pad_value='minimum'): 0.2
            }, p=0.8),
            tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.15),
            tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.15),
            tio.OneOf({
                tio.transforms.RandomNoise(mean=0, std=0.1),
                tio.transforms.RandomBlur(std=0.1)
            }, p=0.15),
            tio.OneOf({
                tio.transforms.RandomBiasField(),
                tio.transforms.RandomMotion(num_transforms=2),
                tio.transforms.RandomGhosting(num_ghosts=2),
                tio.transforms.RandomSwap(),
                tio.transforms.RandomSpike()
            }, p=0.03)
        ])

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.raw_dataset.get(self.volumes[idx], no_crop=False)
        raster = self.transform(entry["raster"])

        return {
            "age": entry["age"], 
            "tbv": entry["tbv"],  
            "voxel_vol": entry["voxel_vol"], 
            "voxels": entry["voxels"],
            "raster": raster
        }
    
class ValDataset(RasterDataset):
    def __init__(self, raw_dataset, pids, verbose=True):
        super().__init__(raw_dataset)

        self.pid_set = set(pids)
        self.volumes = self.raw_dataset.get_indices_from_pids(pids)
        
        if verbose: print("Preprocessing validation data...")
        
        self.preprocessed_rasters = {}
        for idx in self.volumes:
            self.preprocessed_rasters[idx] = self.raw_dataset.get(idx, no_crop=True)
        
        if verbose: print("Done preprocessing validation data.")

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.preprocessed_rasters[self.volumes[idx]]

        return {
            "age": entry["age"], 
            "tbv": entry["tbv"],  
            "voxel_vol": entry["voxel_vol"],
            "voxels": entry["voxels"],
            "raster": entry["raster"]
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
            
            if train_mode == TrainMode.NO_AGE:
                pred_voxels = model(rasters).squeeze().cpu()
            elif train_mode == TrainMode.IN_AGE:
                pred_voxels = model(rasters, ages.to(device)).squeeze().cpu()
                ages = ages.cpu()
            elif train_mode == TrainMode.OUT_AGE:
                predictions = model(rasters).squeeze().cpu()
                pred_voxels, pred_ages = torch.split(predictions, 1, dim=1)
                pred_ages = pred_ages.detach().numpy().reshape(-1, 1)
            
            pred_voxels = pred_voxels.detach().numpy().reshape(-1, 1)
            pred_voxels = raw_dataset.normalize_voxels(pred_voxels, inverse=True)
            
            if train_mode == TrainMode.OUT_AGE:
                pred_ages = raw_dataset.normalize_ages(pred_ages, inverse=True)
                ages = raw_dataset.normalize_ages(ages, inverse=True)
          
            pred_tbvs = pred_voxels * data["voxel_vol"].numpy().reshape(-1, 1)

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
        optimizer_params={}, criterion_params={}, num_epochs=500, patience=100, 
        early_stop_ignore_first_epochs= 100,  batch_size=8, data_workers=4, trace_func=print, 
        scheduler_class=None, scheduler_params={}, k_fold=6, grad_clip=5, override_val_pids=None,
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
        
        train_set = TrainDataset(raw_dataset, train_pids)
        val_set = ValDataset(raw_dataset, val_pids)

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