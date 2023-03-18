import pandas as pd
import numpy as np

import os
import gc
import scipy
import nrrd
import torch
import torchio as tio

from multiprocessing import Pool

from enum import Enum

from torch.utils.data import Dataset, DataLoader


class RawDataset:
    def __init__(self, data_dir, side_len=96, no_crop=False, no_deform=True, crop_factor=0.8, crop_chance=0.3, verbose=True):
        self.data_dir = data_dir
        self.side_len = side_len
        self.no_crop = no_crop
        self.no_deform = no_deform
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

            if self.no_crop:
                tensor, voxel_vol = self._prep_raster(tensor, voxel_vol, no_crop=True)
                voxel_count = tbv / voxel_vol

                self.voxels_min = min(self.voxels_min, voxel_count)
                self.voxels_max = max(self.voxels_max, voxel_count)

                if verbose:
                    print(f"Loaded {pid} with {voxel_count:.2f} voxels. Age: {age}, TBV: {tbv}, Voxel Volume: {voxel_vol:.3f} cm3.")
            else:
                upper_voxels, lower_voxels = self._get_voxel_range(tensor, tbv, voxel_vol)
                voxel_count = (upper_voxels + lower_voxels) / 2

                self.voxels_min = min(self.voxels_min, lower_voxels)
                self.voxels_max = max(self.voxels_max, upper_voxels)

                if verbose:
                    print(f"Loaded {pid} with {lower_voxels:.2f} - {upper_voxels:.2f} voxels. Age: {age}, TBV: {tbv}, Voxel Volume: {voxel_vol:.3f} cm3.")

            self.age_min = min(self.age_min, age)
            self.age_max = max(self.age_max, age)

            data.append({
                "pid": pid,
                "age": age,
                "tbv": tbv,
                "voxel_count": voxel_count,
                "voxel_vol": voxel_vol,
                "raster": tensor
            })

        self.dataset = pd.DataFrame(columns=["pid", "age", "tbv", "voxel_count", "voxel_vol", "raster"], data=data)
        self.dataset[["age", "tbv", "voxel_count", "voxel_vol"]] = self.dataset[["age", "tbv", "voxel_count", "voxel_vol"]].apply(pd.to_numeric)

        self.voxels_mean = self.dataset["voxel_count"].mean()
        self.voxels_std = self.dataset["voxel_count"].std()
        self.age_mean = self.dataset["age"].mean()
        self.age_std = self.dataset["age"].std()

        self.voxels_mean = (self.voxels_mean - self.voxels_min) / (self.voxels_max - self.voxels_min)
        self.voxels_std = self.voxels_std / (self.voxels_max - self.voxels_min)
        self.age_mean = (self.age_mean - self.age_min) / (self.age_max - self.age_min)
        self.age_std = self.age_std / (self.age_max - self.age_min)

        self.dataset["age"] = self.normalize_age(self.dataset["age"])

        # The voxels the brain occupies are calculated on the fly depending on cropping and zooming if cropping is enabled. 
        # 'voxel_count' is then only meant for data standardization, not for prediction. 
        # It is dropped after we have the mean and std for the dataset. 
        if not self.no_crop:
            self.dataset.drop(columns=["voxel_count"], inplace=True)
        else:
            self.dataset["voxel_count"] = self.normalize_voxels(self.dataset["voxel_count"])

        if verbose:
            self.print_stats()
    
    def _get_voxel_range(self, raster, tbv, voxel_vol):
        original_voxels = tbv / voxel_vol

        if self.no_crop: 
            return original_voxels, original_voxels

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
        
    def _prep_raster(self, raster, voxel_vol, no_crop=False):
        if self.no_deform:
            # First pad and optionally crop the raster to be a cube
            raster = self._reshape_raster(raster, no_crop=no_crop)
            original_side_len = raster.shape[-1]
            original_voxel_vol = voxel_vol
            
            # Resize the raster to be a cube of side length new_side_len
            resize_factor = self.side_len / original_side_len

            new_raster = torch.from_numpy(scipy.ndimage.zoom(raster.squeeze(0), (resize_factor, resize_factor, resize_factor), order=1)).unsqueeze(0)
            new_voxel_vol = original_voxel_vol / resize_factor
        else:
            # Without reshaping first, deform the raster into a cube of side length self.side_len
            resize_factors = [self.side_len / raster.shape[1], self.side_len / raster.shape[2], self.side_len / raster.shape[3]]
            new_raster = torch.from_numpy(scipy.ndimage.zoom(raster.squeeze(0), resize_factors, order=1)).unsqueeze(0)
            new_voxel_vol = voxel_vol / np.prod(resize_factors)

        return new_raster, new_voxel_vol

    def __len__(self):
        return len(self.dataset)

    def get(self, idx, no_crop=False):
        no_crop = no_crop or (np.random.uniform() > self.crop_chance)

        item = self.dataset.iloc[idx]
        if self.no_crop:
            raster = item["raster"]
            voxel_count = item["voxel_count"]
            voxel_vol = item["voxel_vol"]
        else:
            raster, voxel_vol = self._prep_raster(item["raster"], item["voxel_vol"], no_crop=no_crop)
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
    class TransformMode(Enum):
        NORMAL = 0
        BAYES = 1
        NONE = 2

    def __init__(self, raw_dataset, pids, verbose=True):
        super().__init__(raw_dataset)

        self.pid_set = set(pids)
        self.volumes = self.raw_dataset.get_indices_from_pids(pids)

        self.mode = 0

        self.train_transform = tio.Compose([
            tio.OneOf({
                tio.transforms.RandomAffine(scales=0, degrees=15, translation=0.1, default_pad_value='minimum'): 0.8,
                tio.transforms.RandomAffine(scales=0, degrees=90, translation=0.1, default_pad_value='minimum'): 0.2
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

        self.bayes_transform = tio.Compose([
            tio.transforms.RandomAffine(scales=0, degrees=10, translation=0, default_pad_value='minimum', p=0.6),
            tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
            tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.2),
            tio.OneOf({
                tio.transforms.RandomNoise(mean=0, std=0.05),
                tio.transforms.RandomBlur(std=0.05)
            }, p=0.3)
        ])

    def set_mode(self, mode):
        self.mode = mode
        
    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.raw_dataset.get(self.volumes[idx], no_crop=False)

        if self.mode == self.TransformMode.NORMAL:   raster = self.train_transform(entry["raster"])
        elif self.mode == self.TransformMode.BAYES:  raster = self.bayes_transform(entry["raster"])
        else:                                        raster = entry["raster"]

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

        train_loss = training_loss/len(tr_dataloader)
        train_loss_list.at[epoch] = train_loss

        if val_dataloader is not None:
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

            val_loss = validation_loss/len(val_dataloader)
            val_loss_list.at[epoch] = val_loss

        if(scheduler):  scheduler.step(train_loss)

        if val_dataloader is not None:
            stopping = early_stopper(epoch, val_loss, model)
            trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Dropout p: {dropout_value:.2f}, Patience: {early_stopper.counter}/{early_stopper.patience}            ")
        else:
            stopping = early_stopper(epoch, train_loss, model)
            trace_func(f"Epoch [{epoch+1:04d}/{num_epochs:04d}], Training Loss: {train_loss:.4f}, Dropout p: {dropout_value:.2f}, Patience: {early_stopper.counter}/{early_stopper.patience}            ")

        if stopping:
            break
    
    return train_loss_list, val_loss_list

class Final_results:
    class Bayes_results:
        def __init__(self, name):
            self.name = name
            self.tbv_error_mean = None
            self.tbv_error_std = None
            
            self.age_error_mean = None
            self.age_error_std = None
            
            self.big_error_mean = None
            self.big_error_std = None
            self.big_error_count = None
            self.big_error_threshold = None

            self.refusal_threshold = None
            self.refused_count = 0

        def set_tbv_error(self, tbv_error_mean, tbv_error_std):
            self.tbv_error_mean = tbv_error_mean
            self.tbv_error_std = tbv_error_std
        
        def set_age_error(self, age_error_mean, age_error_std):
            self.age_error_mean = age_error_mean
            self.age_error_std = age_error_std
        
        def set_big_error(self, big_error_mean, big_error_std, big_error_count, big_error_threshold):
            self.big_error_mean = big_error_mean
            self.big_error_std = big_error_std
            self.big_error_count = big_error_count
            self.big_error_threshold = big_error_threshold
        
        def set_refused_count(self, refused_count, refusal_threshold):
            self.refused_count = refused_count
            self.refusal_threshold = refusal_threshold

        def to_dict(self):
            return {
                "name": self.name,
                "tbv_error_mean": self.tbv_error_mean,
                "tbv_error_std": self.tbv_error_std,
                "age_error_mean": self.age_error_mean,
                "age_error_std": self.age_error_std,
                "big_error_mean": self.big_error_mean,
                "big_error_std": self.big_error_std,
                "big_error_count": self.big_error_count,
                "big_error_threshold": self.big_error_threshold,
                "refused_count": self.refused_count,
                "refusal_threshold": self.refusal_threshold
            }

    def __init__(self):
        self.tbv_error_mean = None
        self.tbv_error_std = None
        
        self.age_error_mean = None
        self.age_error_std = None
        
        self.big_error_mean = None
        self.big_error_std = None
        self.big_error_count = None

        self.total_rasters = 0
        self.bayes_results = []

    def set_tbv_error(self, tbv_error_mean, tbv_error_std):
        self.tbv_error_mean = tbv_error_mean
        self.tbv_error_std = tbv_error_std
    
    def set_age_error(self, age_error_mean, age_error_std):
        self.age_error_mean = age_error_mean
        self.age_error_std = age_error_std
    
    def set_big_error(self, big_error_mean, big_error_std, big_error_count, big_error_threshold):
        self.big_error_mean = big_error_mean
        self.big_error_std = big_error_std
        self.big_error_count = big_error_count
        self.big_error_threshold = big_error_threshold

    def add_bayes_results(self, bayes_results):
        self.bayes_results.append(bayes_results)

    def set_total_rasters(self, total_rasters):
        self.total_rasters = total_rasters

    def to_dict(self):
        return {
            "total_rasters": self.total_rasters,
            "tbv_error_mean": self.tbv_error_mean,
            "tbv_error_std": self.tbv_error_std,
            "age_error_mean": self.age_error_mean,
            "age_error_std": self.age_error_std,
            "big_error_mean": self.big_error_mean,
            "big_error_std": self.big_error_std,
            "big_error_count": self.big_error_count,
            "big_error_threshold": self.big_error_threshold,
            "bayes_results": [bayes_result.to_dict() for bayes_result in self.bayes_results]
        }

def final_eval(model, dataloader, dataset, device, train_mode, bayes_runs=30, max_bayes_mse=9.0, big_diff_threshold=25, trace_func=print, verbose=False):
    all_diffs_tbv = []
    all_diffs_age = []

    if(verbose):
        trace_func("\nEvaluating model on test set...")

    with torch.no_grad():
        dataset.set_mode(TrainDataset.TransformMode.NONE)

        model.eval()
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
            pred_voxels = dataset.normalize_voxels(pred_voxels, inverse=True)
            
            if train_mode == TrainMode.OUT_AGE:
                pred_ages = dataset.normalize_ages(pred_ages, inverse=True)
                ages = dataset.normalize_ages(ages, inverse=True)
          
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

        final_results = Final_results()
        
        all_diffs_tbv = np.concatenate(all_diffs_tbv)
        avg_diff_tbv = round(np.mean(all_diffs_tbv), 2)
        std_diff_tbv = round(np.std(all_diffs_tbv), 2)

        final_results.set_tbv_error(avg_diff_tbv, std_diff_tbv)
        final_results.set_total_rasters(len(all_diffs_tbv))

        all_big_diffs = [diff for diff in all_diffs_tbv if diff > big_diff_threshold]

        if len(all_big_diffs) > 0:
            big_diff_mean = round(np.mean(all_big_diffs), 2)
            big_diff_std = round(np.std(all_big_diffs), 2)
        else:
            big_diff_mean = None
            big_diff_std = None
        
        final_results.set_big_error(big_diff_mean, big_diff_std, len(all_big_diffs), big_diff_threshold)

        if train_mode == TrainMode.OUT_AGE:
            all_diffs_age = np.concatenate(all_diffs_age)
            avg_diff_age = round(np.mean(all_diffs_age), 2)
            std_diff_age = round(np.std(all_diffs_age), 2)

            final_results.set_age_error(avg_diff_age, std_diff_age)
        
        # Now evaluate the model with Bayesian runs in 3 different modes.
        if bayes_runs and bayes_runs > 0:
            trace_func(f"\nEvaluating model with {bayes_runs} Bayesian runs...")

            model.enable_dropblock = False
            for i in range(3):
                # Filtered differences
                bayes_diffs_tbv = []
                bayes_diffs_age = []

                # Raw differences
                bayes_matrix_tbv = []
                bayes_matrix_age = []

                refused_raster_count = 0
                
                if i == 0:
                    # Dropout + transforms
                    model.train()
                    dataset.set_mode(TrainDataset.TransformMode.BAYES)
                    adjusted_max_bayes_mse = max_bayes_mse * 1.2 # 1.15 better for folding
                    name = "Dropout + Transforms Bayes"
                    trace_func(f"\nEvaluating model with transforms and dropout...")
                elif i == 1:
                    # Transforms only
                    model.eval()
                    dataset.set_mode(TrainDataset.TransformMode.BAYES)
                    adjusted_max_bayes_mse = max_bayes_mse * 1.0 # 0.625
                    name = "Transforms Bayes"
                    trace_func(f"\nEvaluating model with transforms only...")
                elif i == 2:
                    # Dropout only
                    model.train()
                    dataset.set_mode(TrainDataset.TransformMode.NONE)
                    adjusted_max_bayes_mse = max_bayes_mse * 1.0
                    name = "Dropout Bayes"
                    trace_func(f"\nEvaluating model with dropout only...")

                for _ in range(bayes_runs):
                    run_tbvs = []
                    run_ages = []

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
                        pred_voxels = dataset.normalize_voxels(pred_voxels, inverse=True)

                        if train_mode == TrainMode.OUT_AGE:
                            pred_ages = dataset.normalize_ages(pred_ages, inverse=True)
                            ages = dataset.normalize_ages(ages, inverse=True)

                        run_tbvs.append(pred_voxels * data["voxel_vol"].numpy().reshape(-1, 1))

                        if train_mode == TrainMode.OUT_AGE:
                            run_ages.append(pred_ages)

                    bayes_matrix_tbv.append(np.array(np.concatenate(run_tbvs)).squeeze())
                    if train_mode == TrainMode.OUT_AGE:
                        bayes_matrix_age.append(np.array(np.concatenate(run_ages)).squeeze())

                error = np.std(np.stack(bayes_matrix_tbv), axis=0)

                for i in range(len(error)):
                    if error[i] < adjusted_max_bayes_mse:
                        bayes_diffs_tbv.append(all_diffs_tbv[i][0])
                        if train_mode == TrainMode.OUT_AGE:
                            bayes_diffs_age.append(all_diffs_age[i][0])
                    else:
                        refused_raster_count += 1
                        if(verbose):
                            trace_func(f"Refused raster with bayesian error {error[i]:.2f}. TBV error: {all_diffs_tbv[i][0]:.2f}.")
                
                bayes_results = Final_results.Bayes_results(name=name)
                bayes_results.set_refused_count(refused_raster_count, adjusted_max_bayes_mse)
                if len(bayes_diffs_tbv) > 0:
                    
                    bayes_avg_diff_tbv = round(np.mean(bayes_diffs_tbv), 2)
                    bayes_std_diff_tbv = round(np.std(bayes_diffs_tbv), 2)
                    bayes_results.set_tbv_error(bayes_avg_diff_tbv, bayes_std_diff_tbv)
                    
                    if train_mode == TrainMode.OUT_AGE:
                        bayes_diffs_age = np.array(bayes_diffs_age)
                        bayes_avg_diff_age = round(np.mean(bayes_diffs_age), 2)
                        bayes_std_diff_age = round(np.std(bayes_diffs_age), 2)

                        bayes_results.set_age_error(bayes_avg_diff_age, bayes_std_diff_age)
                    
                    bayes_big_diffs = [diff for diff in bayes_diffs_tbv if diff > big_diff_threshold]

                    if len(bayes_big_diffs) > 0:
                        big_diff_mean = round(np.mean(bayes_big_diffs), 2)
                        big_diff_std = round(np.std(bayes_big_diffs), 2)
                    else:
                        big_diff_mean = None
                        big_diff_std = None

                    bayes_results.set_big_error(big_diff_mean, big_diff_std, len(bayes_big_diffs), big_diff_threshold)

                final_results.add_bayes_results(bayes_results)
    return final_results


def trace_results(final_results, trace_func):
    trace_func()
    trace_func("- - - - - -")
    trace_func("Non-bayesian prediction:")
    trace_func(f"Total Raster Count: {final_results.total_rasters}")
    trace_func(f"Mean Absolute Error: {final_results.tbv_error_mean:.2f} cc")
    trace_func(f"Standard Deviation: {final_results.tbv_error_std:.2f} cc")
    if final_results.age_error_mean:
        trace_func(f"Mean Absolute Error Age: {final_results.age_error_mean:.2f} days")
        trace_func(f"Standard Deviation Age: {final_results.age_error_std:.2f} days")
    trace_func(f"Big error count (>{final_results.big_error_threshold}): {final_results.big_error_count}")
    if(final_results.big_error_count > 0):
        trace_func(f"Big error mean: {final_results.big_error_mean:.2f} cc")
        trace_func(f"Big error std: {final_results.big_error_std:.2f} cc")
    trace_func("- - - - - -")
    
    for bres in final_results.bayes_results: 
        trace_func()
        trace_func(f"Bayesian prediction {bres.name}:")

        trace_func(f"Refused Raster Count: {bres.refused_count}")
        trace_func(f"Refused Raster Percentage: {bres.refused_count / final_results.total_rasters * 100:.2f}%")
        trace_func(f"Refusal threshold: {bres.refusal_threshold}")
        if bres.tbv_error_mean:
            trace_func(f"Mean Absolute Error: {bres.tbv_error_mean:.2f} cc")
            trace_func(f"Standard Deviation: {bres.tbv_error_std:.2f} cc")
            if bres.age_error_mean:
                trace_func(f"Mean Absolute Error Age: {bres.age_error_mean:.2f} days")
                trace_func(f"Standard Deviation Age: {bres.age_error_std:.2f} days")
            trace_func(f"Big error count (>{bres.big_error_threshold}): {bres.big_error_count}")
            if bres.big_error_count > 0:
                trace_func(f"Big error mean: {bres.big_error_mean:.2f} cc")
                trace_func(f"Big error std: {bres.big_error_std:.2f} cc")
        else:
            trace_func("No rasters were accepted.")
    trace_func("- - - - - -")
    trace_func()
    
def run_folds(model, raw_dataset, device, optimizer_class, criterion_class, train_mode, 
        optimizer_params={}, criterion_params={}, num_epochs=500, patience=100, 
        early_stop_ignore_first_epochs= 100,  batch_size=8, data_workers=4, trace_func=print, 
        scheduler_class=None, scheduler_params={}, k_fold=6, grad_clip=5, override_val_pids=None,
        dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0),
        bayes_runs=30, max_bayes_mse=9.0):

    # Save model to reset it after each fold
    torch.save(model.state_dict(), "base_weights.pt")

    train_score = []
    val_score = []
    final_results = []
    
    unique_pids = raw_dataset.get_unique_pids()
    total_pids = len(unique_pids)
    fraction = 1/k_fold
    seg = int(total_pids*fraction)
    
    # tr:train, val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trace_func(f" - - - Fold {i+1}/{k_fold} - - -")
        trace_func()

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

        trace_func(f"Train volume count: {len(train_set)}")
        trace_func(f"Validation volume count: {len(val_set)}")
        trace_func(f"Validation volumes: {train_pids}")
        trace_func()

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=data_workers, pin_memory=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=data_workers, pin_memory=True)
        
        train_set.set_mode(TrainDataset.TransformMode.NORMAL)
        train_loss_list, val_loss_list = train(
            model, criterion, optimizer, train_dataloader, val_dataloader, early_stopper, 
            train_mode, num_epochs, device, trace_func=trace_func, scheduler=scheduler,
            dropout_change=dropout_change, dropout_change_epochs=dropout_change_epochs, dropout_range=dropout_range, 
            grad_clip=grad_clip
        )
        
        model.load_state_dict(torch.load("best_weights.pt"))

        train_score.append(train_loss_list)
        val_score.append(val_loss_list)
        final_results.append(
            final_eval(model, val_dataloader, raw_dataset, device, train_mode, 
                       bayes_runs=bayes_runs, max_bayes_mse=max_bayes_mse,
                       trace_func=trace_func, verbose=True)
        )

        trace_results(final_results[-1], trace_func=trace_func)

        os.rename("best_weights.pt", f"best_weights_{i+1}.pt")

        if override_val_pids:
            # If we are overriding the validation set, we don't want to do any more folds
            break

    if os.path.exists("best_weights.pt"):
        os.remove("best_weights.pt")
    if os.path.exists("base_weights.pt"):
        os.remove("base_weights.pt")
    
    return train_score, val_score, final_results

def run_final(model, raw_dataset, device, optimizer_class, criterion_class, train_mode, 
        optimizer_params={}, criterion_params={}, num_epochs=500, patience=100, 
        early_stop_ignore_first_epochs= 100, batch_size=8, data_workers=4, trace_func=print, 
        scheduler_class=None, scheduler_params={}, grad_clip=5,
        dropout_change=0.0, dropout_change_epochs=10, dropout_range=(0.0, 1.0),
        bayes_runs=30, max_bayes_mse=9.0, eval_only=False):

    train_score = []
    final_results = []
    
    all_pids = raw_dataset.get_unique_pids()
    
    dataset = TrainDataset(raw_dataset, all_pids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=data_workers, pin_memory=True)

    trace_func(f"Train volume count: {len(dataloader)}")
    trace_func()
    
    if not eval_only:
        criterion = criterion_class(**criterion_params)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = None if not scheduler_class else scheduler_class(optimizer, **scheduler_params)
        early_stopper = EarlyStopper(
            patience=patience, ignore_first_epochs=early_stop_ignore_first_epochs, 
            verbose=False, path="best_weights.pt", trace_func=trace_func
        )

        dataset.set_mode(TrainDataset.TransformMode.NORMAL)
        train_score, _ = train(
            model, criterion, optimizer, dataloader, None, early_stopper, 
            train_mode, num_epochs, device, trace_func=trace_func, scheduler=scheduler,
            dropout_change=dropout_change, dropout_change_epochs=dropout_change_epochs, dropout_range=dropout_range, 
            grad_clip=grad_clip
        )
    
        model.load_state_dict(torch.load("best_weights.pt"))

    final_results = final_eval(model, dataloader, dataset, device, train_mode, 
                    bayes_runs=bayes_runs, max_bayes_mse=max_bayes_mse,
                    trace_func=trace_func, verbose=True)
    
    trace_results(final_results, trace_func=trace_func)

    if os.path.exists("best_weights.pt"):
        os.rename("best_weights.pt", "final_weights.pt")
    
    return train_score, final_results