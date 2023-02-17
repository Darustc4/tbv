import numpy as np
import torch
import torch.nn as nn

from models.shared import RawDataset, ValDataset
from torch.utils.data import DataLoader

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
    raw_dataset = RawDataset(data_dir="./dataset")
    model = RasterNet(ResidualBlock, [3, 3, 3, 3])

    model.load_state_dict(torch.load("test_bayesian.pt", map_location=torch.device('cpu')))

    val_set = ValDataset(raw_dataset, ['23', '48', '38', '1', '80', '22', '27', '36'])
    test_dataloader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=8)
    
    all_accepted_diffs = []
    all_diffs = []
    
    refused_raster_count = 0
    model.train()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            rasters = data["raster"].float()
            tbvs = data["tbv"].float().numpy().reshape(-1, 1)

            # Inject a fake random raster
            #rasters[0] = torch.rand(rasters[0].shape)
            
            all_predictions = []
            for _ in range(10):
                predictions = model(rasters).squeeze()
                all_predictions.append(predictions.detach().numpy())
            
            all_predictions = np.stack(all_predictions)

            predictions = np.mean(all_predictions, axis=0).reshape(-1, 1)
            error = np.std(all_predictions, axis=0).reshape(-1, 1)

            # Revert the normalization and standardization
            predictions = raw_dataset.voxels_std.inverse_transform(predictions)
            predictions = raw_dataset.voxels_minmax.inverse_transform(predictions)
            predicted_tbvs = predictions * data["voxel_volume"].numpy().reshape(-1, 1)

            tbv_diffs = np.abs(predicted_tbvs - tbvs)

            accepted_diffs = []
            for j in range(len(tbvs)):
                if error[j][0] < 0.1:
                    accepted_diffs.append(tbv_diffs[j][0])
                else:
                    refused_raster_count += 1

            all_accepted_diffs.append(accepted_diffs)
            all_diffs.append(tbv_diffs)     

    all_accepted_diffs = np.concatenate(all_accepted_diffs)
    all_diffs = np.concatenate(all_diffs)

    print(f"Refused Raster Count: {refused_raster_count}")
    print(f"Total Raster Count: {len(test_dataloader.dataset)}")
    print()
    print("Among accepted rasters:")
    print(f"Mean Absolute Error: {np.mean(all_accepted_diffs):.2f} cc")
    print(f"Standard Deviation: {np.std(all_accepted_diffs):.2f} cc")
    print()
    print("Among all rasters:")
    print(f"Mean Absolute Error: {np.mean(all_diffs):.2f} cc")
    print(f"Standard Deviation: {np.std(all_diffs):.2f} cc")
    