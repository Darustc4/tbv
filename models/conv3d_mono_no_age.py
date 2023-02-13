import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from shared import *

cuda = torch.device('cuda')

class RasterNet(nn.Module):
    def __init__(self):
        super(RasterNet, self).__init__()

        self.conv0 = nn.Sequential( # 1x96x96x96 -> 32x48x48x48
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential( # 32x48x48x48 -> 64x24x24x24
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential( # 64x24x24x24 -> 128x12x12x12
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential( # 128x12x12x12 -> 256x6x6x6
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential( # 256x6x6x6 -> 512x3x3x3
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.conv5 = nn.Sequential( # 512x3x3x3 -> 1024x1x1x1
            nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(1024),
            nn.ReLU()
        )

        self.linear = nn.Sequential( 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.to(cuda)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

if __name__ == "__main__":
    dataset = RasterDataset(data_dir="../dataset")
    model = RasterNet().to(cuda)

    tr_score, val_score, unscaled_loss = cross_validator(
        model, dataset, cuda, k_fold=6, num_epochs=1000, patience=50, 
        optimizer_class=torch.optim.Adam, criterion_class=nn.MSELoss, 
        learning_rate=0.0001, weight_decay=0.00001, batch_size=4, data_workers=4,
        verbose=True, trace_func=print, fold_limit=None
    )
    """
    # first fold training and validation loss plot
    plt.plot(tr_score[0], label="Training Loss", linewidth=1.5)
    plt.plot(val_score[0], label="Validation Loss", linewidth=1.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/conv3d_mono_no_age_train.png")


    print("Average Loss: ", unscaled_loss[0][0])
    print("Standard Deviation: ", unscaled_loss[0][1])

    """
    figure, axis = plt.subplots(3, 2)
    for i in range(3):
        for j in range(2):
            axis[i][j].plot(tr_score[i*2+j], label="Training Loss", linewidth=1.5)
            axis[i][j].plot(val_score[i*2+j], label="Validation Loss", linewidth=1.5)
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
    