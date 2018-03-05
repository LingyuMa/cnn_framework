class ProjectType:
    classification = 0
    detection = 1
    segmentation = 2


class ImageType:
    rgb = 0
    gray = 1
    binary = 2
    rgba = 3


class LabelType:
    text = 0
    image = 1


class Normalization:
    none = 0
    symmetric = 1
    positive = 2


class RotationType:
    none = 0
    limit_90 = 1
    limit_180 = 2


class FlipType:
    none = 0
    up_down = 1
    left_right = 2
    combined = 3


class ShiftType:
    none = 0
    width_shift = 1
    height_shift = 2
    combined = 3


class NoiseType:
    none = 0
    gaussian_noise = 1
    poisson_noise = 2
    speckle_noise = 3


class Optimizer:
    Adam = 0
    SGD = 1

