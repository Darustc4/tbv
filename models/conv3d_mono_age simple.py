import matplotlib.pyplot as plt

import torch
import torch.nn as nn

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

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(13824, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)

        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    dataset = RasterDataset(data_dir="../dataset")
    model = RasterNet().to(cuda)
    
    print(f"Model has {model.count_parameters()} trainable parameters")

    tr_score, val_score, unscaled_loss = run(
        model, dataset, cuda, optimizer_class=torch.optim.SGD, criterion_class=nn.MSELoss, 
        train_fun=train_tbv_age, final_eval_fun=final_eval_tbv_age, 
        optimizer_params={"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001, "nesterov": True}, criterion_params={},
        k_fold=6, num_epochs=2000, patience=250,
        batch_size=8, data_workers=8, trace_func=print,
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau, 
        scheduler_params={"mode": "min", "factor": 0.5, "patience": 10, "threshold": 0.0001, "verbose": True},
        override_val_pids=['23', '48', '38', '1', '80', '22', '27', '36']
    )
    
    # first fold training and validation loss plot
    plt.plot(tr_score[0], label="Training Loss", linewidth=1.0)
    plt.plot(val_score[0], label="Validation Loss", linewidth=1.0)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig("plots/conv3d_mono_age_single_train.png")