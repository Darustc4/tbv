import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from shared import *

torch.manual_seed(0)
np.random.seed(0)
cuda = torch.device('cuda')

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_ch = out_ch
        
    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.conv1(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class RasterNet(nn.Module):
    def __init__(self, block, layers):
        super(RasterNet, self).__init__()
        self.inplanes = 64

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.layer0 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.do = nn.Dropout(p=0.0)
        
        self.fc0 = nn.Linear(512, 1024)
        self.fc1 = nn.Linear(1024, 1)

        self.enable_dropblock = True


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.layer0(x)
        if self.enable_dropblock: x = ops.drop_block3d(x, block_size=3, p=self.do.p, training=self.training)
        x = self.layer1(x)
        if self.enable_dropblock: x = ops.drop_block3d(x, block_size=3, p=self.do.p, training=self.training)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.do(x)
        x = F.relu(self.fc0(x))
        x = self.do(x)
        x = self.fc1(x)
        
        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    raw_dataset = RawDataset(data_dir="../../dataset/original", side_len=96, no_crop=True, no_deform=False)
    model = RasterNet(ResidualBlock, [3, 3, 5, 4]).to(cuda)
    
    print(f"Model has {model.count_parameters()} trainable parameters")

    with open("plots/conv3d_no_age_resnet_log.txt", "w") as f:
        
        def logger(file):
            def log_to_file(msg="", end="\n"):
                print(msg, end=end)
                if end != '\r':
                    file.write(msg + end)
            return log_to_file

        tr_score, val_score, final_results = run_folds(
            model, raw_dataset, cuda, optimizer_class=torch.optim.SGD, criterion_class=nn.MSELoss, train_mode=TrainMode.NO_AGE,
            optimizer_params={"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}, criterion_params={},
            k_fold=6, num_epochs=300, patience=75, early_stop_ignore_first_epochs=100, 
            batch_size=8, data_workers=8, trace_func=logger(f),
            dropout_change=0, dropout_change_epochs=1, dropout_range=(0.3, 0.3),
            scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"mode": "min", "factor": 0.4, "patience": 10, "threshold": 0.0001, "verbose": True},
            bayes_runs=30, max_bayes_mse=9.0,
            #override_val_pids=['23', '48', '38', '1', '80', '22', '27', '36']
        )
        
        # first fold training and validation loss plot
        plt.plot(tr_score[0], label="Training Loss", linewidth=1.0)
        plt.plot(val_score[0], label="Validation Loss", linewidth=1.0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.savefig("plots/conv3d_no_age_resnet.png")

        # Move best weights and plot to folders
        if not os.path.exists("weights"):
            os.makedirs("weights")
        if not os.path.exists("plots"):
            os.makedirs("plots")
        
        label_std_params = {
            "voxels_min": raw_dataset.voxels_min,
            "voxels_max": raw_dataset.voxels_max,
            "voxels_mean": raw_dataset.voxels_mean,
            "voxels_std": raw_dataset.voxels_std
        }

        with open(os.path.join("weights", "conv3d_no_age_resnet.json"), "w") as f:
            json.dump(label_std_params, f, indent=2)
        
        final_stats = [r.to_dict() for r in final_results]
        with open(os.path.join("plots", "conv3d_no_age_resnet_stats.json"), "w") as f:
            json.dump(final_stats, f, indent=2)
        
        for i in range(len(final_results)):
            os.rename(f"best_weights_{i+1}.pt", f"weights/conv3d_no_age_resnet_{i+1}.pt")

        if len(final_results) == 6:
            plt.clf()
            figure, axis = plt.subplots(3, 2)
            for i in range(3):
                for j in range(2):
                    axis[i][j].plot(tr_score[i*2+j], label="Training Loss", linewidth=0.7)
                    axis[i][j].plot(val_score[i*2+j], label="Validation Loss", linewidth=0.7)
                    axis[i][j].set_xlabel("Epochs")
                    axis[i][j].set_ylabel("Loss")

            handles, labels = axis[i][j].get_legend_handles_labels()
            figure.legend(handles, labels, loc='lower center')

            figure.tight_layout()
            plt.savefig("plots/conv3d_no_age_resnet_folds.png", dpi=300)


            bayes0 = [fold["bayes_results"][0] for fold in final_stats]
            bayes1 = [fold["bayes_results"][1] for fold in final_stats]
            bayes2 = [fold["bayes_results"][2] for fold in final_stats]
            for fold in final_stats:
                del fold["bayes_results"]

            no_filtering_stats = pd.DataFrame.from_records([r.to_dict() for r in final_results])
            bayes0_stats = pd.DataFrame.from_records(bayes0)
            bayes1_stats = pd.DataFrame.from_records(bayes1)
            bayes2_stats = pd.DataFrame.from_records(bayes2)

            plt.clf()
            figure, axis = plt.subplots()
            
            trans0 = Affine2D().translate(-0.15, 0.0) + axis.transData
            trans1 = Affine2D().translate(-0.05, 0.0) + axis.transData
            trans2 = Affine2D().translate(0.05, 0.0) + axis.transData
            trans3 = Affine2D().translate(0.15, 0.0) + axis.transData
            
            axis.errorbar(x=range(1,7), y=no_filtering_stats["tbv_error_mean"], yerr=no_filtering_stats["tbv_error_std"], fmt='o', label="No filtering", transform=trans0)
            axis.errorbar(x=range(1,7), y=bayes0_stats["tbv_error_mean"], yerr=bayes0_stats["tbv_error_std"], fmt='o', label=bayes0[0]["name"], transform=trans1)
            axis.errorbar(x=range(1,7), y=bayes1_stats["tbv_error_mean"], yerr=bayes1_stats["tbv_error_std"], fmt='o', label=bayes1[0]["name"], transform=trans2)
            axis.errorbar(x=range(1,7), y=bayes2_stats["tbv_error_mean"], yerr=bayes2_stats["tbv_error_std"], fmt='o', label=bayes2[0]["name"], transform=trans3)
            
            axis.set_xlabel("Fold")
            axis.set_ylabel("TBV Loss")
            figure.legend()
            figure.savefig("plots/conv3d_no_age_resnet_folds_bayes.png")

            