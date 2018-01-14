import pandas as pd

from typedef import *


class Label:
    def __init__(self, label_path, project_type=ProjectType.classification):
        self.project_type = project_type
        self.label_path = label_path
        self.lookup = None
        self.load_label_lookup()

    def load_label_lookup(self):
        self.lookup = pd.read_csv(self.label_path, sep=',')

    def get(self, image_name):
        if self.project_type == ProjectType.classification:
            return classification_label_loader(image_name, self.lookup)
        elif self.project_type == ProjectType.detection:
            raise NotImplementedError('object detection not implemented')
        elif self.project_type == ProjectType.segmentation:
            raise NotImplementedError('object segmentation not implemented')
        else:
            raise ValueError('undefined project type')


def classification_label_loader(image_name, lookup_df):
    return lookup_df[image_name]
