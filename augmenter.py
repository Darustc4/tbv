import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy
import nrrd
import argparse
import random
import multiprocessing
import tqdm

from torchvision import transforms
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class Pipeline:
    transform_count = 8

    def __init__(self, target_size, force_fingerprint=None):
        trfs = []
        fingerprint = [False for _ in range(Pipeline.transform_count)] if force_fingerprint is None else force_fingerprint

        # Crop
        if force_fingerprint is None: fingerprint[0] = random.choice([True, False])
        if fingerprint[0]:
            trfs.append(Pipeline.FixedCrop(np.random.randint(int(0.9*target_size), target_size)))

        # Affine
        if force_fingerprint is None: fingerprint[1] = random.choice([True, False])
        if fingerprint[1]:
            translation = tuple(np.random.uniform(0.0, 0.3, 2))
            angle = np.random.uniform(-180., 180.)
            scale = np.random.uniform(0.8, 1.2)
            shear = np.random.uniform(-0.2, 0.2)
            trfs.append(Pipeline.FixedAffine(translation, angle, scale, shear, 0.))

        # Flip
        if force_fingerprint is None: fingerprint[2:4] = np.random.uniform(0., 1., 2) < 0.5
        if fingerprint[2:4] != [False, False]:
            trfs.append(Pipeline.FixedFlip(fingerprint[2], fingerprint[3]))

        # Noise
        if force_fingerprint is None: fingerprint[4:6] = np.random.uniform(0., 1., 3) < 0.2
        if fingerprint[4:6] != [False, False]:
            trfs.append(Pipeline.RandomNoise(*fingerprint[4:6]))

        # Sharpness
        if force_fingerprint is None: fingerprint[6] = random.choice([True, False])
        if fingerprint[6]:
            trfs.append(Pipeline.FixedSharpness(np.random.uniform(0.8, 1.2)))

        # Brightness
        if force_fingerprint is None: fingerprint[7] = random.choice([True, False])
        if fingerprint[7]:
            trfs.append(Pipeline.FixedBrightness(np.random.uniform(0.7, 1.3)))

        np.random.shuffle(trfs)

        self.pipeline = transforms.Compose(trfs)
        self.fingerprint = tuple(fingerprint)

    def __call__(self, img):
        return self.pipeline(img)

    def __key(self):
        return self.fingerprint

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Pipeline):
            return self.__key() == __o.__key()
        return False

    class FixedCrop:
        def __init__(self, size):
            self.size = size

        def __call__(self, img):
            return transforms.functional.resized_crop(img, img.size[0] - self.size, img.size[1] - self.size, self.size, self.size, img.size, interpolation=Image.BILINEAR)

    class FixedAffine:
        def __init__(self, translate, angle, scale, shear, fill):
            self.translate = translate
            self.angle = angle
            self.scale = scale
            self.shear = shear
            self.fill = fill

        def __call__(self, img):
            return transforms.functional.affine(img, translate=self.translate, angle=self.angle, scale=self.scale, shear=self.shear, fill=self.fill)

    class FixedFlip:
        def __init__(self, flip_h, flip_v):
            self.flip_h = flip_h
            self.flip_v = flip_v

        def __call__(self, img):
            img = transforms.functional.hflip(img) if self.flip_h else img
            img = transforms.functional.vflip(img) if self.flip_v else img
            return img

    class FixedBrightness:
        def __init__(self, brightness):
            self.brightness = brightness

        def __call__(self, img):
            return transforms.functional.adjust_brightness(img, self.brightness)

    class FixedSharpness:
        def __init__(self, sharpness):
            self.sharpness = sharpness

        def __call__(self, img):
            return transforms.functional.adjust_sharpness(img, self.sharpness)

    class RandomNoise:
        def __init__(self, sp, gauss):
            self.sp = sp
            self.gauss = gauss

            self.gauss_range = 20
            self.sp_thresh = 0.005

        def __call__(self, img):
            img = np.array(img)

            if self.sp:
                noise = np.random.rand(img.shape[0], img.shape[1])
                img[noise >= (1-self.sp_thresh)] = np.random.randint(200, 255)
                img[noise <= self.sp_thresh] = np.random.randint(0, 50)
            if self.gauss:
                img = img + np.random.randint(-self.gauss_range/2, self.gauss_range/2, (img.shape[0],img.shape[1]), dtype=np.int8)
                img = np.clip(img, 0, 255)

            return Image.fromarray(np.uint8(img))


class Augmenter:
    class Datapoint:
        def __init__(self, path, pid, age, tbv):
            self.path = path
            self.pid = pid
            self.age = age
            self.tbv = tbv

    def __init__(self, data_path, output_path, target_size, augmentation_factor):
        self.expected_meta_columns = {'id', 'birthdate', 'scandate', 'tbv'}

        self.data_path = data_path
        self.output_path = output_path
        self.target_size = target_size
        self.augmentation_factor = augmentation_factor

        self._setup()

    def augment(self):
        total_augmentations = len(self.dataset) * self.augmentation_factor

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()*2)

        print(f"Processing {len(self.dataset)} 3D raster images with {self.augmentation_factor} augmentations each.")
        for _ in tqdm.tqdm(pool.imap_unordered(self._job, self.dataset), total=len(self.dataset)):
            pass

        print(f"Finished augmenting for a total of {total_augmentations} images.")

    def _job(self, datapoint):
        data, header = nrrd.read(datapoint.path)
        original_shape = data.shape

        resize_factors = [self.target_size/original_shape[0], self.target_size/original_shape[1], self.target_size/original_shape[2]]

        data = scipy.ndimage.zoom(data, resize_factors)
        new_shape = data.shape
        header['sizes'] = data.shape

        # Each image will be processed by up to 3 pipelines, one for each axis
        # Force the first augmentation to be the original image
        augmentation_pipelines = [(None, None, None)]

        for _ in range(self.augmentation_factor-1):
            x_pipe = Pipeline(self.target_size) if random.choice([True, False]) else None
            y_pipe = Pipeline(self.target_size) if random.choice([True, False]) else None
            z_pipe = Pipeline(self.target_size) if random.choice([True, False]) else None

            if x_pipe is None and y_pipe is None and z_pipe is None:
                augmentation_pipelines.append([Pipeline(self.target_size), Pipeline(self.target_size), Pipeline(self.target_size)])
            else:
                augmentation_pipelines.append([x_pipe, y_pipe, z_pipe])

        # Apply the augmentations to the image
        for id, pipelines in enumerate(augmentation_pipelines):
            new_data = np.copy(data)

            for axis in range(3):
                pipeline = pipelines[axis]

                if pipeline is None:
                    continue

                for slice in range(new_shape[axis]):
                    if axis == 0:    new_data[slice, :, :] = np.array(pipeline(Image.fromarray(new_data[slice, :, :])))
                    elif axis == 1:  new_data[:, slice, :] = np.array(pipeline(Image.fromarray(new_data[:, slice, :])))
                    else:            new_data[:, :, slice] = np.array(pipeline(Image.fromarray(new_data[:, :, slice])))

            if id == 0: tbv = datapoint.tbv
            else:       tbv = datapoint.tbv + np.random.uniform(-2., 2.) # Add a bit of noise to better simulate real data

            new_path = os.path.join(self.output_path, f"aug_{datapoint.pid}_{datapoint.age}_{tbv:.2f}_{id}.nrrd")
            nrrd.write(new_path, new_data, header=header)

    def _setup(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.dataset = self._generate_dataset()

    def _generate_dataset(self):
        data_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.data_path) for f in filenames if os.path.splitext(f)[1] == '.nrrd']

        dataset = []
        for data_path in data_paths:
            try:
                filename, ext = os.path.splitext(os.path.basename(data_path))
                split_filename = filename.split("_")
                patient_id = split_filename[1]
                age = int(split_filename[2])
                measured_tbv = float(split_filename[3])
                dataset.append(Augmenter.Datapoint(data_path, patient_id, age, measured_tbv))
            except Exception as e:
                print(f"Could not parse filename {filename}. Skipping file.")
                continue

        return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int, help="Size of the output raster image. (128 is recommended)")
    parser.add_argument('factor', type=int, help="Number of augmented raster images per original raster image")

    parser.add_argument('-d', '--data', type=str, default='dataset')
    parser.add_argument('-o', '--output', type=str, default='aug_dataset')
    args = parser.parse_args()

    if args.size <= 0 or args.factor <= 0:
        raise Exception("The size and factor must be positive integers.")

    if args.factor > 50:
        raise Exception("The maximum factor is 50.")

    augmenter = Augmenter(args.data, args.output, args.size, args.factor)
    augmenter.augment()
