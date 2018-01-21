import numpy as np
import pandas as pd

from skimage.io import imread

from typedef import *


class Label:
    def __init__(self, label_path, project_type=ProjectType.classification):
        self.project_type = project_type
        self.label_path = label_path
        self._Label = None
        if project_type == ProjectType.classification:
            self._Label = ClassificationLabel(label_path)
        elif project_type == ProjectType.detection:
            self._Label = DetectionLabel(label_path)
        elif project_type == ProjectType.segmentation:
            self._Label = SegmentationLabel(label_path)
        else:
            raise NotImplementedError("unknown project type")

    def get(self, image_name):
        return self._Label.get(image_name)


class ClassificationLabel:
    def __init__(self, label_path):
        self.label_path = label_path
        self.lookup = {}
        self.label_sheet = {}
        self.load_label_lookup()

    def load_label_lookup(self):
        label_df = pd.read_csv(self.label_path, sep=',', header=None)
        self.label_sheet = {category: i for i, category in enumerate(label_df[1].unique())}
        for _, row in label_df.iterrows():
            self.lookup[row[0]] = self.one_hot_coding(self.label_sheet[row[1]], len(self.label_sheet))

    def get(self, image_name):
        return self.lookup[image_name]

    @staticmethod
    def one_hot_coding(label_id, label_size):
        res = np.zeros(label_size)
        res[label_id] = 1
        return res


class SegmentationLabel:
    def __init__(self, label_path):
        self.label_path = label_path
        self.lookup = {}
        self.load_label_lookup()

    def load_label_lookup(self):
        label_df = pd.read_csv(self.label_path, sep=',', header=None)
        for _, row in label_df.iterrows():
            self.lookup[row[0]] = row[1]

    def get(self, image_name):
        return imread(self.lookup[image_name], as_grey=True)


class DetectionLabel:
    def __init__(self, label_path):
        pass
    pass
