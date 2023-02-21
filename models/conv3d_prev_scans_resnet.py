import matplotlib.pyplot as plt

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
        
        self.fc0 = nn.Linear(515, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)

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

    def forward(self, raster, age, prev_scans):
        x = self.conv0(raster)
        x = self.layer0(x)
        if self.enable_dropblock: x = ops.drop_block3d(x, block_size=3, p=self.do.p, training=self.training)
        x = self.layer1(x)
        if self.enable_dropblock: x = ops.drop_block3d(x, block_size=3, p=self.do.p, training=self.training)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.do(x)
        x = torch.cat((x, age, prev_scans), dim=1)
        x = F.relu(self.fc0(x))
        x = self.do(x)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = self.fc2(x)

        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    raw_dataset = RawDataset(data_dir="../dataset", compute_prev_scans=True, prev_scans_n=1, prev_scans_no_age=False, prev_scans_age_as_diff=False)
    model = RasterNet(ResidualBlock, [3, 3, 4, 4]).to(cuda)
    
    # We only need to train the last layers from scratch
    model_dict = torch.load("weights/conv3d_no_age_resnet_best.pt")
    model_dict = {k: v for k, v in model_dict.items() if not k.startswith("fc")}
    model.load_state_dict(model_dict, strict=False)

    print(f"Model has {model.count_parameters()} trainable parameters")

    tr_score, val_score, unscaled_loss = run(
        model, raw_dataset, cuda, optimizer_class=torch.optim.SGD, criterion_class=nn.MSELoss, train_mode=TrainMode.PREV_SCANS,
        optimizer_params={"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}, criterion_params={},
        k_fold=6, num_epochs=500, patience=50, early_stop_ignore_first_epochs=25, 
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
        "voxels_min": raw_dataset.voxels_minmax.data_min_.item(),
        "voxels_max": raw_dataset.voxels_minmax.data_max_.item(),
        "voxels_mean": raw_dataset.voxels_std.mean_.item(),
        "voxels_std": raw_dataset.voxels_std.scale_.item()
    }

    with open(os.path.join("weights", "conv3d_prev_scans_resnet.json"), "w") as f:
        json.dump(label_std_params, f)
    plt.savefig("plots/conv3d_prev_scans_resnet.png")
    os.rename("best_weights.pt", "weights/conv3d_prev_scans_resnet.pt")
    