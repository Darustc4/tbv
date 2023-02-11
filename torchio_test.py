import torchio as tio
import torch
import nrrd

# Read the image
image, headers = nrrd.read('./dataset/1_01-02-2018.nrrd')
tensor = torch.from_numpy(image).unsqueeze(0)
image = tio.ScalarImage(tensor=tensor)

transform = tio.Compose([
    tio.transforms.RandomAffine(scales=0, degrees=180, translation=0.3, default_pad_value='minimum', p=0.8),
    tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
    tio.transforms.RandomAnisotropy(axes=(0, 1, 2), p=0.3),
    tio.transforms.RandomNoise(mean=0, std=0.1, p=0.3),
    tio.transforms.RandomBlur(std=0.1, p=0.3),
    tio.transforms.RandomBiasField(coefficients=0.5, order=3, p=0.3)
])

transformed = transform(image)
nrrd.write('./aug_dataset/1_01-02-2018_augmented.nrrd', transformed.data.numpy()[0], headers)
