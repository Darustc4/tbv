import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from shared import *

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

        self.conv0 = nn.Sequential( # 1x96x96x96 -> 64x48x48x48
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.layer0 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(13824, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
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
        x = x.to(cuda)
        x = self.conv0(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)

        return x

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    raw_dataset = RawDataset(data_dir="../dataset")
    model = RasterNet(ResidualBlock, [3, 3, 3, 3]).to(cuda)
    
    print(f"Model has {model.count_parameters()} trainable parameters")

    tr_score, val_score, unscaled_loss = run(
        model, raw_dataset, cuda, optimizer_class=torch.optim.SGD, criterion_class=nn.MSELoss, 
        train_fun=train_tbv, final_eval_fun=final_eval_tbv, 
        optimizer_params={"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001, "nesterov": True}, criterion_params={},
        k_fold=6, num_epochs=1500, patience=200,
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
    plt.savefig("plots/conv3d_mono_no_age_single_train.png")