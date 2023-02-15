# Models

This project predicts TBV and age of patients with 4 different main architectures:
- Simple and shallow 3D ConvNet.
- Deep 3D ResNet.
- Feature engineered histogram preprocessing + Simple ConvNet.
- Bayesian 3D Net.

Validating on: ['23', '48', '38', '1', '80', '22', '27', '36']

```
tio.transforms.RandomAffine(scales=0, degrees=10, translation=0.2, default_pad_value='minimum', p=0.25),
tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.2),
tio.transforms.RandomNoise(mean=0, std=0.1, p=0.2),
tio.transforms.RandomBlur(std=0.1, p=0.2),
tio.transforms.RandomBiasField(p=0.01),
tio.transforms.RandomMotion(num_transforms=2, p=0.01),
tio.transforms.RandomGhosting(num_ghosts=2, p=0.01),
tio.transforms.RandomSwap(p=0.01),
tio.transforms.RandomSpike(p=0.01)
```

---
# 3D Convolution No Age

3D convolutions and linear perceptron. Age of the patient is not used.

## No Residual

Model has 19396225 trainable parameters

1500 epochs. Validation patience of 200 epochs. Batch size 8.

Criterion: MSE Loss
Optimizer: SGD. LR 0.001. WD 0.0001. Mo 0.9. Nesterov.
Scheduler: ReduceLROnPlateau. Mode min. Factor 0.5. Patience 10. Threshold 0.0001.

Results OLD:
- Average difference: 13.39 cc.   
- Standard deviation: 13.15 cc.

## ResNet [3, 3, 3, 3]

Model has 66657601 trainable parameters.

1500 epochs. Validation patience of 200 epochs. Batch size 8.

Criterion: MSE Loss
Optimizer: SGD. LR 0.001. WD 0.0001. Mo 0.9. Nesterov.
Scheduler: ReduceLROnPlateau. Mode min. Factor 0.5. Patience 10. Threshold 0.0001.

Results OLD:
- Average difference: 15.1 cc.    
- Standard deviation: 16.74 cc.

---
# 3D Convolution Age

3D convolutions and linear perceptron. Age of the patient is also predicted.

## No Residual

Model has 19396738 trainable parameters.

1500 epochs. Validation patience of 200 epochs. Batch size 8.

Criterion: MSE Loss
Optimizer: SGD. LR 0.001. WD 0.0001. Mo 0.9. Nesterov.
Scheduler: ReduceLROnPlateau. Mode min. Factor 0.5. Patience 10. Threshold 0.0001.

Results OLD:
- Average difference TBV: 13.69 cc.	
- Standard deviation TBV: 13.87 cc.
- Average difference Age: 9.71 days.	
- Standard deviation Age: 9.26 days.

## ResNet [3, 3, 3, 3]

Model has 66658114 trainable parameters.

1500 epochs. Validation patience of 200 epochs. Batch size 8.

Criterion: MSE Loss
Optimizer: SGD. LR 0.001. WD 0.0001. Mo 0.9. Nesterov.
Scheduler: ReduceLROnPlateau. Mode min. Factor 0.5. Patience 10. Threshold 0.0001.

Results OLD:
- Average difference TBV: 14.15 cc.       
- Standard deviation TBV: 15.64 cc.
- Average difference Age: 9.89 days. 
- Standard deviation Age: 10.14 days.


---

# Other Experiments


## Resnet Age [1,1,1,1]

Results:


## Resnet NoAge [3,3,3,3]
