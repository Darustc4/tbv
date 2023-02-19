import json
import os
import nrrd
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
    def __init__(self, data_dir, pids, std_params={}, use_age=False, output_noise=False):
        self.data_dir = data_dir
        self.pids = pids
        self.use_age = use_age
        self.output_noise = output_noise

        try:
            self.histogram_bins = std_params["histogram_bins"]
            self.drop_zero_bin = std_params["drop_zero_bin"]
            if self.drop_zero_bin: self.histogram_bins += 1
        except KeyError as e:
            self.histogram_bins = None
            self.drop_zero_bin = False

        self.voxels_min = std_params["voxels_min"]
        self.voxels_max = std_params["voxels_max"]
        self.voxels_mean = std_params["voxels_mean"]
        self.voxels_std = std_params["voxels_std"]
        if self.use_age:
            self.age_min = std_params["age_min"]
            self.age_max = std_params["age_max"]
            self.age_mean = std_params["age_mean"]
            self.age_std = std_params["age_std"]

        self.intensity_transform = tio.transforms.RescaleIntensity()
        self.znorm_transform = tio.transforms.ZNormalization()

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
                tensor = self._get_histogram(tensor) if self.histogram_bins else tensor
                tensor = self.intensity_transform(tensor)
                tensor = self.znorm_transform(tensor)

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

        self.dataset["voxels"] = self.normalize_voxels(self.dataset["voxels"])
        if self.use_age:
            self.dataset["age"] = self.normalize_age(self.dataset["age"])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.dataset.iloc[idx]
        
        item = {
            "pid": entry["pid"], 
            "tbv": entry["tbv"], 
            "voxels": entry["voxels"], 
            "voxel_volume": entry["voxel_volume"], 
            "raster": entry["raster"]
        }

        if self.use_age:
            item["age"] = entry["age"]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest="model_name", help="Name of the model to test (simple, resnet, hist, bayes)")
    parser.add_argument("--use_age", action="store_true", help="Use age in the model")
    parser.add_argument("--skip_bayesian", action="store_true", help="Skip bayesian dropout")
    parser.add_argument("--use_latest", action="store_true", help="Use latest weights instead of best")
    parser.add_argument("--mode", type=str, default="lenient", help="Refuse raster option (lenient, normal, strict)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--bayes_runs", type=float, default=10, help="Number of times to estimate each sample")
    parser.add_argument("--data_dir", type=str, default="../dataset", help="Path to the dataset directory")
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
        dataset = Dataset(data_dir=args.data_dir, pids=pids, std_params=std_params, output_noise=args.noise)

    elif args.model_name == "resnet":
        from conv3d_no_age_resnet import RasterNet, ResidualBlock
        model = RasterNet(ResidualBlock, [3, 3, 4, 4])
        dataset = Dataset(data_dir=args.data_dir, pids=pids, std_params=std_params, output_noise=args.noise)

    elif args.model_name == "hist":
        from conv3d_no_age_hist import RasterNet, ResidualBlock
        model = RasterNet(ResidualBlock, [2, 2, 2, 2])
        dataset = Dataset(data_dir=args.data_dir, pids=pids, std_params=std_params, output_noise=args.noise)

    elif args.model_name == "bayes":
        pass
        #from conv3d_no_age_bayesian import RasterNet
    else:
        raise ValueError("Invalid model name")

    if args.mode == "lenient":     refuse_threshold = 0.2
    elif args.mode == "normal":    refuse_threshold = 0.12
    elif args.mode == "strict":    refuse_threshold = 0.08
    else: raise ValueError("Invalid mode")

    model.load_state_dict(torch.load(os.path.join(args.weights_dir, weights_file), map_location=torch.device('cpu')))
    model.enable_dropblock = False
    model.do.p = args.dropout

    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print("Computing the mean and standard deviation of the test set for non-bayesian mode...")

    eval_all_diffs = []
    eval_all_tbvs = []

    model.eval()
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
        
        refused_raster_count = 0

        model.train()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                rasters = data["raster"].float()
                tbvs = data["tbv"].float().numpy().reshape(-1, 1)

                all_predictions = []
                for _ in range(args.bayes_runs):
                    predictions = model(rasters).squeeze()
                    all_predictions.append(predictions.detach().numpy())
                
                all_predictions = np.stack(all_predictions)

                predictions = np.mean(all_predictions, axis=0).reshape(-1, 1)
                error = np.std(all_predictions, axis=0).reshape(-1, 1)

                predictions = dataset.normalize_voxels(predictions, inverse=True)
                predicted_tbvs = predictions * data["voxel_volume"].numpy().reshape(-1, 1)

                #predicted_tbvs = (predicted_tbvs + eval_all_tbvs[i * batch_size:(i + 1) * batch_size]) / 2 # Option 1
                #predicted_tbvs = eval_all_tbvs[i * batch_size:(i + 1) * batch_size]                        # Option 2
                predicted_tbvs = predicted_tbvs                                                            # Option 3
                
                tbv_diffs = np.abs(predicted_tbvs - tbvs)

                accepted_diffs = []
                for j in range(len(tbvs)):
                    if error[j][0] < refuse_threshold:
                        accepted_diffs.append(tbv_diffs[j][0])
                    else:
                        refused_raster_count += 1

                filtered_diffs.append(accepted_diffs)
                bayes_all_diffs.append(tbv_diffs)     

        filtered_diffs = np.concatenate(filtered_diffs)
        bayes_all_diffs = np.concatenate(bayes_all_diffs)

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
        print()
        print("Among all rasters:")
        print(f"Mean Absolute Error: {np.mean(bayes_all_diffs):.2f} cc")
        print(f"Standard Deviation: {np.std(bayes_all_diffs):.2f} cc")
        print(f"Big error count (>{big_diff_threshold}): {len(bayes_all_big_diffs)}")
        if (len(bayes_all_big_diffs) > 0):
            print(f"Big error mean: {np.mean(bayes_all_big_diffs):.2f} cc")
            print(f"Big error std: {np.std(bayes_all_big_diffs):.2f} cc")
        print("- - - - - -")
    
    