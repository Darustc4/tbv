import matplotlib.pyplot as plt

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import *

torch.manual_seed(0)
np.random.seed(0)
cuda = torch.device('cuda')

class RasterNet(nn.Module):
    def __init__(self):
        super(RasterNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.do = nn.Dropout(0.0)

        self.fc0 = nn.Linear(1025, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1)
    
    def forward(self, raster, age):
        x = self.conv0(raster)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, age), dim=1)

        x = self.do(x)
        x = F.relu(self.fc0(x))
        x = self.do(x)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)

        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    raw_dataset = RawDataset(data_dir="../../dataset/original", side_len=96, crop_factor=0.8)
    model = RasterNet().to(cuda)
    
    print(f"Model has {model.count_parameters()} trainable parameters")

    tr_score, val_score, unscaled_loss = run(
        model, raw_dataset, cuda, optimizer_class=torch.optim.SGD, criterion_class=nn.MSELoss, train_mode=TrainMode.IN_AGE,
        optimizer_params={"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}, criterion_params={},
        k_fold=6, num_epochs=1000, patience=100, early_stop_ignore_first_epochs=125,
        batch_size=8, data_workers=8, trace_func=print,
        dropout_change=0, dropout_change_epochs=1, dropout_range=(0.3, 0.3),
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_params={"mode": "min", "factor": 0.4, "patience": 10, "threshold": 0.0001, "verbose": True},
        override_val_pids=['23', '48', '38', '1', '80', '22', '27', '36']
    )

    # first fold training and validation loss plot
    plt.plot(tr_score[0], label="Training Loss", linewidth=1.0)
    plt.plot(val_score[0], label="Validation Loss", linewidth=1.0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()

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

    with open(os.path.join("weights", "conv3d_in_age_simple.json"), "w") as f:
        json.dump(label_std_params, f)
    plt.savefig("plots/conv3d_in_age_simple.png")
    os.rename("best_weights.pt", "weights/conv3d_in_age_simple.pt")

    """
    figure, axis = plt.subplots(3, 2)
    for i in range(3):
        for j in range(2):
            axis[i][j].plot(tr_score[i*2+j], label="Training Loss", linewidth=1.0)
            axis[i][j].plot(val_score[i*2+j], label="Validation Loss", linewidth=1.0)
            axis[i][j].set_xlabel("Epochs")
            axis[i][j].set_ylabel("Loss")

    handles, labels = axis[i][j].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center')

    figure.tight_layout()
    plt.savefig("plots/conv3d_mono_no_age_train.png")


    unscaled_loss = pd.DataFrame(unscaled_loss.tolist(), columns=['avg','std'], index=unscaled_loss.index)
    avg_loss = unscaled_loss['avg']
    std_loss = unscaled_loss['std']

    plt.clf()
    plt.errorbar(x=range(6), y=avg_loss, yerr=std_loss, fmt='o')
    plt.xlabel("Fold")
    plt.ylabel("TBV Loss")
    plt.savefig("plots/conv3d_mono_no_age_loss.png")
    """