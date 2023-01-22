import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    transform_count = 9

    def __init__(self, target_size, force_fingerprint=None):
        trfs = []
        fingerprint = [False for _ in range(Pipeline.transform_count)] if force_fingerprint is None else force_fingerprint

        # Crop
        if force_fingerprint is None: fingerprint[0] = random.choice([True, False])
        if fingerprint[0]:
            trfs.append(Pipeline.FixedCrop(np.random.randint(int(0.85*target_size), target_size)))

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
        if force_fingerprint is None: fingerprint[4:7] = np.random.uniform(0., 1., 3) < 0.3
        if fingerprint[4:7] != [False, False, False]:
            trfs.append(Pipeline.RandomNoise(*fingerprint[4:7]))

        # Sharpness
        if force_fingerprint is None: fingerprint[7] = random.choice([True, False])
        if fingerprint[7]:
            trfs.append(Pipeline.FixedSharpness(np.random.uniform(0.8, 1.2)))

        # Brightness
        if force_fingerprint is None: fingerprint[8] = random.choice([True, False])
        if fingerprint[8]:
            trfs.append(Pipeline.FixedBrightness(np.random.uniform(0.7, 1.3)))

        np.random.shuffle(trfs)

        self.pipeline = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
            *trfs
        ])

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
            return transforms.functional.resized_crop(img, img.size[0] - self.size, img.size[1] - self.size, self.size, self.size, img.size, interpolation=Image.BICUBIC)

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
        def __init__(self, sp, gauss, poisson):
            self.sp = sp
            self.gauss = gauss
            self.poisson = poisson

            self.gauss_range = 30
            self.sp_thresh = 0.01
            self.poisson_peak = 30

        def __call__(self, img):
            img = np.array(img)

            if self.sp:
                noise = np.random.rand(img.shape[0], img.shape[1])
                img[noise >= (1-self.sp_thresh)] = np.random.randint(200, 255)
                img[noise <= self.sp_thresh] = np.random.randint(0, 50)
            if self.gauss:
                img = img + np.random.randint(-self.gauss_range/2, self.gauss_range/2, (img.shape[0],img.shape[1]))
                img = np.clip(img, 0, 255)
            if self.poisson:
                img = np.random.poisson(img / 255.0 * self.poisson_peak) / self.poisson_peak * 255
                img = np.clip(img, 0, 255)

            return Image.fromarray(np.uint8(img))


class Augmenter:
    class Datapoint:
        def __init__(self, path, pid, age, tbv):
            self.path = path
            self.pid = pid
            self.age = age
            self.tbv = tbv

    def __init__(self, data_path, labels_path, output_path, target_size, augmentation_factor):
        self.expected_meta_columns = {'id', 'birthdate', 'scandate', 'tbv'}

        self.data_path = data_path
        self.labels_path = labels_path
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
        header['sizes'] = np.array([self.target_size, header['sizes'][1], self.target_size]) # New size of the .nrrd file
        array_shape = data.shape                                                             # Original size of the .nrrd file

        # Force the first augmentation to be the original image
        augmentation_pipelines = set([Pipeline(self.target_size, force_fingerprint=[False for _ in range(Pipeline.transform_count)])])
        while len(augmentation_pipelines) < self.augmentation_factor:
            augmentation_pipelines.add(Pipeline(self.target_size))

        # Apply the augmentations to the image
        for id, pipeline in enumerate(augmentation_pipelines):
            new_data = np.zeros(header['sizes'])
            for slice in range(array_shape[1]): # Slice on the coronal axis

                pixel_array = data[:, slice, :]
                pixel_array = np.array(pipeline(Image.fromarray(pixel_array)))
                new_data[:, slice, :] = pixel_array

            new_path = os.path.join(self.output_path, f"aug_{datapoint.pid}_{datapoint.age}_{datapoint.tbv}_{id}.nrrd")
            nrrd.write(new_path, new_data, header=header)

    def _setup(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.labels = self._load_labels()
        self.data_paths = self._get_data_paths()
        self.dataset = self._generate_dataset()

    def _load_labels(self):
        try:
            labels = pd.read_csv(self.labels_path)
        except Exception as e:
            raise Exception("CSV labels file not found or can not be opened.")

        if not self.expected_meta_columns.issubset(set(labels.columns)):
            raise Exception("The metadata file does not contain the columns 'id', 'birthdate', 'scandate' and 'tbv'.")

        return labels

    def _get_data_paths(self):
        dataset = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.data_path) for f in filenames if os.path.splitext(f)[1] == '.nrrd']

        return dataset

    def _generate_dataset(self):
        dataset = []
        for data_path in self.data_paths:
            try:
                filename, _ = os.path.splitext(os.path.basename(data_path))
                split_filename = filename.split("_")
                pid = split_filename[-1]
                scan_date = datetime.datetime(year=(int)(split_filename[0]), month=(int)(split_filename[1]), day=(int)(split_filename[2]))
            except Exception as e:
                raise Exception("The NRRD file must be named following the pattern '<year>_<month>_<day>_<id>'.")

            string_scan_date = scan_date.strftime("%d/%m/%Y")              # IMPORTANT: Format the scan date to match the format in the metadata file
            entry = self.labels[(self.labels["id"] == (int)(pid)) & (self.labels["scandate"] == string_scan_date)]

            if entry.empty:
                raise Exception(f"The metadata file does not contain an entry for the patient with id '{pid}' and scan date '{string_scan_date}'.")

            entry = entry.iloc[0]
            string_birth_date = entry["birthdate"]
            birth_date = datetime.datetime.strptime(string_birth_date, "%d/%m/%Y") # Again, format accordingly
            age = (scan_date - birth_date).days
            tbv = entry["tbv"]

            dataset.append(Augmenter.Datapoint(data_path, pid, age, tbv))

        return dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int, help="Size of the output raster image. (128 is recommended)")
    parser.add_argument('factor', type=int, help="Number of augmented raster images per original raster image")

    parser.add_argument('--data', type=str, default='dataset')
    parser.add_argument('--labels', type=str, default='dataset/labels.csv')
    parser.add_argument('--output', type=str, default='aug_dataset')
    args = parser.parse_args()

    if args.size <= 0 or args.factor <= 0:
        raise Exception("The size and factor must be positive integers.")

    if args.factor > 50:
        raise Exception("The maximum factor is 50.")

    augmenter = Augmenter(args.data, args.labels, args.output, args.size, args.factor)
    augmenter.augment()
