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
    symmetric = 0
    positive = 1
    none = 2


class RotationType:
    limit_90 = 0
    limit_180 = 1
    none = 2


class FlipType:
    up_down = 0
    left_right = 1
    combined = 2
    none = 3


class ShiftType:
    width_shift = 0
    height_shift = 1
    combined = 2
    none = 3


class NoiseType:
    gaussian_noise = 0
    poisson_noise = 1
    speckle_noise = 2
    none = 3
