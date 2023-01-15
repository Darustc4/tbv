import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import nrrd
import argparse

class Augmenter:
    class Datapoint:
        def __init__(self, path, pid, age, tbv):
            self.path = path
            self.pid = pid
            self.age = age
            self.tbv = tbv

    def __init__(self, data_path, labels_path, output_path):
        self.expected_meta_columns = {'id', 'birthdate', 'scandate', 'tbv'}

        self.data_path = data_path
        self.labels_path = labels_path
        self.output_path = output_path

        self._setup()

    def augment(self):
        for datapoint in self.dataset:
            data, header = nrrd.read(datapoint.path)
            array_shape = data.shape

            """
            for augmentation in augmentation_list:
                for i in array_shape[1]:
                    pixel_array = data[:, i, :]
            """

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
    parser.add_argument('--data', type=str, default='dataset')
    parser.add_argument('--labels', type=str, default='dataset/labels.csv')
    parser.add_argument('--output', type=str, default='aug_dataset')
    args = parser.parse_args()

    augmenter = Augmenter(args.data, args.labels, args.output)
    augmenter.augment()
