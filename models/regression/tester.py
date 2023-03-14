import json
import os
import nrrd
import scipy
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader
from numba import jit

torch.manual_seed(0)
np.random.seed(0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, pids, side_len=96, no_crop=False, no_deform=True, crop_factor=0.8, crop_chance=0.3, std_params={}, use_age=False, output_noise=False):
        self.data_dir = data_dir
        self.pids = pids
        self.side_len = side_len
        self.no_crop = no_crop
        self.no_deform = no_deform
        self.crop_factor = crop_factor
        self.crop_chance = crop_chance
        self.use_age = use_age
        self.output_noise = output_noise

        intensity_transform = tio.transforms.RescaleIntensity()
        znorm_transform = tio.transforms.ZNormalization()

        self.voxels_min = std_params["voxels_min"]
        self.voxels_max = std_params["voxels_max"]
        self.voxels_mean = std_params["voxels_mean"]
        self.voxels_std = std_params["voxels_std"]
        if self.use_age:
            self.age_min = std_params["age_min"]
            self.age_max = std_params["age_max"]
            self.age_mean = std_params["age_mean"]
            self.age_std = std_params["age_std"]

        data = []
        for file in os.listdir(data_dir):
            # Split filename by _ and get the first element, which is the pid
            pid = file.split("_")[0]
            if pid in pids:
                if self.output_noise:
                    headers = nrrd.read_header(os.path.join(data_dir, file))
                    raster = np.random.randint(0, 255, headers["sizes"])
                else:
                    raster, headers = nrrd.read(os.path.join(data_dir, file))

                tensor = torch.from_numpy(raster).unsqueeze(0)
                tensor = intensity_transform(tensor)
                tensor = znorm_transform(tensor)

                voxel_volume = np.prod(headers["spacings"])
                tbv = float(headers["tbv"])
                age = int(headers["age_days"])

                if self.no_crop:
                    tensor, voxel_volume = self._prep_raster(tensor, voxel_volume, no_crop=True)
                    voxel_count = tbv / voxel_volume

                data.append({
                    "pid": pid,
                    "age": age,
                    "tbv": tbv,
                    "voxel_volume": voxel_volume,
                    "voxels": voxel_count if self.no_crop else None,
                    "raster": tensor
                })

        self.dataset = pd.DataFrame(columns=["pid", "age", "tbv", "voxel_volume", "voxels", "raster"], data=data)
        self.dataset[["age", "tbv", "voxel_volume", "voxels"]] = self.dataset[["age", "tbv", "voxel_volume", "voxels"]].apply(pd.to_numeric)

        if self.use_age:
            self.dataset["age"] = self.normalize_age(self.dataset["age"])

    def __len__(self):
        return len(self.dataset)
        
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

    def set_no_crop(self, no_crop):
        self.no_crop = no_crop

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]

        if not self.no_crop:
            no_crop = np.random.uniform() > self.crop_chance

            raster, voxel_volume = self._prep_raster(item["raster"], item["voxel_volume"], no_crop=no_crop)
            voxel_count = self.normalize_voxels(item["tbv"] / voxel_volume)
        else:
            raster = item["raster"]
            voxel_volume = item["voxel_volume"]
            voxel_count = self.normalize_voxels(item["voxels"])

        item = {
            "age": item["age"],
            "tbv": item["tbv"],
            "voxel_volume": voxel_volume,
            "voxels": voxel_count,
            "raster": raster
        }

        return item

    def normalize_voxels(self, voxels, inverse=False):
        if not inverse:
            voxels = (voxels - self.voxels_min) / (self.voxels_max - self.voxels_min)
            voxels = (voxels - self.voxels_mean) / self.voxels_std
        else:
            voxels = voxels * self.voxels_std + self.voxels_mean
            voxels = voxels * (self.voxels_max - self.voxels_min) + self.voxels_min

        return voxels
    
    def normalize_age(self, age, inverse=False):
        if not inverse:
            age = (age - self.age_min) / (self.age_max - self.age_min)
            age = (age - self.age_mean) / self.age_std
        else:
            age = age * self.age_std + self.age_mean
            age = age * (self.age_max - self.age_min) + self.age_min

        return age


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest="model_name", help="Name of the model to test (simple, resnet)")
    parser.add_argument("--use_age", action="store_true", help="Use age in the model")
    parser.add_argument("--skip_bayesian", action="store_true", help="Skip bayesian dropout")
    parser.add_argument("--use_latest", action="store_true", help="Use latest weights instead of best")
    parser.add_argument("--mode", type=str, default="lenient", help="Refuse raster option (lenient, normal, strict)")
    parser.add_argument("--bayes_runs", type=int, default=10, help="Number of times to estimate each sample")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to the dataset directory")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Path to the weights directory")
    parser.add_argument("--noise", action="store_true", help="Use noise instead of proper rasters")

    args = parser.parse_args()

    print("Loading model and dataset...")
    
    weights_file = f"conv3d_{'no_' if not args.use_age else ''}age_{args.model_name}{'_best' if not args.use_latest else ''}.pt"
    params_file = f"conv3d_{'no_' if not args.use_age else ''}age_{args.model_name}{'_best' if not args.use_latest else ''}.json"

    pids = ['23', '48', '38', '1', '80', '22', '27', '36']
    with open(os.path.join(args.weights_dir, params_file)) as f:
        std_params = json.load(f)

    if args.model_name == "simple":
        from conv3d_no_age_simple import RasterNet
        model = RasterNet()
    elif args.model_name == "resnet":
        from conv3d_no_age_resnet import RasterNet, ResidualBlock
        model = RasterNet(ResidualBlock, [3, 3, 5, 4])
        model.do.p = 0.3
        model.enable_dropblock = False
    else:
        raise ValueError("Invalid model name")

    dataset = Dataset(
        data_dir=args.data_dir, pids=pids, std_params=std_params, output_noise=args.noise, 
        no_crop=True, no_deform=False, side_len=96, crop_factor=0.9, crop_chance=1.0
    )

    if args.mode == "lenient":     refuse_threshold = 12.0
    elif args.mode == "normal":    refuse_threshold = 9.0
    elif args.mode == "strict":    refuse_threshold = 6.0
    else: raise ValueError("Invalid mode")

    model.load_state_dict(torch.load(os.path.join(args.weights_dir, weights_file), map_location=torch.device('cpu')))
    
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Computing the mean and standard deviation of the test set for non-bayesian mode...")

    eval_all_diffs = []
    eval_all_tbvs = []
    transforms = tio.Compose([
        tio.transforms.RandomAffine(scales=0, degrees=10, translation=0, default_pad_value='minimum', p=0.6),
        tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
        tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.2),
        tio.OneOf({
            tio.transforms.RandomNoise(mean=0, std=0.05),
            tio.transforms.RandomBlur(std=0.05)
        }, p=0.3)
    ])

    model.eval()
    dataset.set_no_crop(True)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            rasters = data["raster"].float()
            tbvs = data["tbv"].float().numpy().reshape(-1, 1)
            
            predictions = model(rasters).squeeze().reshape(-1, 1)
            predictions = dataset.normalize_voxels(predictions, inverse=True)
            predicted_tbvs = predictions * data["voxel_volume"].numpy().reshape(-1, 1)

            tbv_diffs = np.abs(predicted_tbvs - tbvs)

            eval_all_tbvs.append(predicted_tbvs)
            eval_all_diffs.append(tbv_diffs)     

    eval_all_diffs = np.concatenate(eval_all_diffs)
    eval_all_tbvs = np.concatenate(eval_all_tbvs)

    if not args.skip_bayesian:
        print("Computing the mean and standard deviation of the test set for bayesian mode...")

        filtered_diffs = []
        bayes_all_diffs = []
        
        diff_matrix = []

        refused_raster_count = 0

        model.train()
        dataset.set_no_crop(False)
        with torch.no_grad():
            for _ in range(args.bayes_runs):
                run_diffs = []
                for i, data in enumerate(dataloader):
                    rasters = data["raster"].float()
                    tbvs = data["tbv"].float().numpy().reshape(-1, 1)
                    
                    trf_rasters = torch.zeros(rasters.shape)
                    for j in range(rasters.shape[0]):
                        trf_rasters[j] = transforms(rasters[j])
                    
                    predictions = model(trf_rasters).squeeze().reshape(-1, 1)
                    predictions = dataset.normalize_voxels(predictions, inverse=True)
                    predicted_tbvs = predictions * data["voxel_volume"].numpy().reshape(-1, 1)
                    run_diffs.append(predicted_tbvs)

                diff_matrix.append(np.array(np.concatenate(run_diffs)).squeeze())

            error = np.std(np.stack(diff_matrix), axis=0)

            accepted_diffs = []
            for i in range(len(error)):
                if error[i] < refuse_threshold:
                    accepted_diffs.append(eval_all_diffs[i][0])
                else:
                    refused_raster_count += 1

            filtered_diffs = np.array(accepted_diffs)

    big_diff_threshold = 30
    eval_all_big_diffs = [diff for diff in eval_all_diffs if diff > big_diff_threshold]
    if not args.skip_bayesian: 
        bayes_all_big_diffs = [diff for diff in bayes_all_diffs if diff > big_diff_threshold]
        bayes_all_accepted_big_diffs = [diff for diff in filtered_diffs if diff > big_diff_threshold]

    print("- - - - - -")
    print(f"Total Raster Count: {len(dataset)}")
    if not args.skip_bayesian: print(f"Refused Raster Count: {refused_raster_count}")
    print("- - - - - -")
    print()
    print("- - - - - -")
    print("Non-bayesian prediction:")
    print(f"Mean Absolute Error: {np.mean(eval_all_diffs):.2f} cc")
    print(f"Standard Deviation: {np.std(eval_all_diffs):.2f} cc")
    print(f"Big error count (>{big_diff_threshold}): {len(eval_all_big_diffs)}")
    if(len(eval_all_big_diffs) > 0):
        print(f"Big error mean: {np.mean(eval_all_big_diffs):.2f} cc")
        print(f"Big error std: {np.std(eval_all_big_diffs):.2f} cc")
    print("- - - - - -")
    if not args.skip_bayesian:
        print()
        print("- - - - - -")
        print("Bayesian prediction:")
        print("Among accepted rasters:")
        print(f"Mean Absolute Error: {np.mean(filtered_diffs):.2f} cc")
        print(f"Standard Deviation: {np.std(filtered_diffs):.2f} cc")
        print(f"Big error count (>{big_diff_threshold}): {len(bayes_all_accepted_big_diffs)}")
        if(len(bayes_all_accepted_big_diffs) > 0):
            print(f"Big error mean: {np.mean(bayes_all_accepted_big_diffs):.2f} cc")
            print(f"Big error std: {np.std(bayes_all_accepted_big_diffs):.2f} cc")
        print("- - - - - -")
    
    