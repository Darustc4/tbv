import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import nrrd
import torch
import torch.nn as nn
import torchio as tio

from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset, DataLoader


class RasterDataset(Dataset):
    def __init__(self, data_dir):
        self.validation = False

        self.data_dir = data_dir

        intensity_transform = tio.transforms.RescaleIntensity()
        znorm_transform = tio.transforms.ZNormalization()

        data = []
        for file in os.listdir(data_dir):
            image, headers = nrrd.read(os.path.join(data_dir, file))

            tensor = torch.from_numpy(image).unsqueeze(0)
            tensor = intensity_transform(tensor)
            tensor = znorm_transform(tensor)

            data.append({
                "pid": os.path.splitext(file)[0].split("_")[1],
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

        self.transform = tio.Compose([
            tio.transforms.RandomAffine(scales=0, degrees=180, translation=0.3, default_pad_value='minimum', p=0.8),
            tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.3),
            tio.transforms.RandomNoise(mean=0, std=0.1, p=0.3),
            tio.transforms.RandomBlur(std=0.1, p=0.3),
            tio.transforms.RandomBiasField(coefficients=0.5, order=3, p=0.3)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.dataset.iloc[idx]
        raster = entry["raster"] if self.validation else self.transform(entry["raster"])

        return {"pid": entry["pid"], "age": entry["age"], "tbv": entry["tbv"], "voxels": entry["voxels"], "voxel_volume": entry["voxel_volume"], "raster": raster}

    def get_unique_pids(self):
        return self.dataset["pid"].unique()
    
    def get_indices_from_pids(self, pids):
        return self.dataset[self.dataset["pid"].isin(pids)].index
        
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

def train(model, criterion, optimizer, tr_dataloader, val_dataloader, early_stopper, num_epochs, device, trace_func=print):
    train_loss_list = pd.Series(dtype=np.float32)
    val_loss_list = pd.Series(dtype=np.float32)

    for epoch in range(num_epochs):
        training_loss = 0.
        
        model.train()
        for i, data in enumerate(tr_dataloader):
            rasters = data["raster"].float().to(device)
            voxels = data["voxels"].float().to(device)

            optimizer.zero_grad()

            predictions = model(rasters).squeeze()
            loss = criterion(predictions, voxels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if (i+1) % 10 == 0:
                trace_func(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(tr_dataloader)}], Loss: {loss.item():.4f}              ", end="\r")

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

        trace_func(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}               ")
        if early_stopper(validation_loss, model):
            break
    
    return train_loss_list, val_loss_list

def get_unscaled_loss(model, criterion, dataloader, dataset):
    all_diffs = []
    for i, data in enumerate(dataloader):
        rasters = data["raster"].float()
        voxels = data["voxels"].float()

        predictions = model(rasters).squeeze().cpu()
        predictions = predictions.detach().numpy().reshape(-1, 1)
        voxels = voxels.detach().numpy().reshape(-1, 1)
        
        # Revert the normalization and standardization
        predictions = dataset.voxels_std.inverse_transform(predictions)
        predictions = dataset.voxels_minmax.inverse_transform(predictions)
        predicted_tbvs = voxels * data["voxel_volume"].numpy().reshape(-1, 1)
        tbvs = data["tbv"].numpy().reshape(-1, 1)

        all_diffs.append(np.abs(predicted_tbvs - tbvs))

    all_diffs = np.concatenate(all_diffs)
    
    avg_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)

    return avg_diff, std_diff

def cross_validator(model, dataset, device, k_fold=5, num_epochs=400, patience=30, optimizer_class=torch.optim.Adam, criterion_class=nn.MSELoss, learning_rate=0.001, weight_decay = 0.00001, batch_size=32, data_workers=4, verbose=False, trace_func=print, fold_limit=None):
    # Save model to reset it after each fold
    torch.save(model.state_dict(), "base_weights.pt")

    train_score = pd.Series(dtype=np.float32)
    val_score = pd.Series(dtype=np.float32)
    unscaled_loss = pd.Series(dtype=np.float32)
    
    unique_pids = dataset.get_unique_pids()
    total_pids = len(unique_pids)
    fraction = 1/k_fold
    seg = int(total_pids*fraction)
    
    # tr:train, val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        if fold_limit is not None and i >= fold_limit:
            break

        trace_func(f"Fold {i+1}/{k_fold}")

        model.load_state_dict(torch.load("base_weights.pt"))

        criterion = criterion_class()
        optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopper = EarlyStopper(patience=patience, verbose=verbose, path="best_weights.pt", trace_func=trace_func)

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

        train_indices = dataset.get_indices_from_pids(unique_pids[train_indices])
        val_indices = dataset.get_indices_from_pids(unique_pids[val_indices])

        print(f"Train volumes: {len(train_indices)}")
        print(f"Validation volumes: {len(val_indices)}")

        train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset, val_indices)
        val_set.validation = True

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=data_workers, drop_last=True)
        test_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=data_workers, drop_last=True)
        
        train_loss_list, val_loss_list = train(model, criterion, optimizer, train_dataloader, test_dataloader, early_stopper, num_epochs, device, trace_func)
        
        model.load_state_dict(torch.load("best_weights.pt"))

        train_score.at[i] = train_loss_list
        val_score.at[i] = val_loss_list
        unscaled_loss.at[i] = get_unscaled_loss(model, criterion, test_dataloader, dataset)
    
    os.remove("base_weights.pt")
    os.remove("best_weights.pt")
    
    return train_score, val_score, unscaled_loss