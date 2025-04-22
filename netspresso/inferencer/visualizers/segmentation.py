import cv2
import numpy as np
from matplotlib import pyplot as plt

from netspresso.inferencer.visualizers.utils import voc_color_map


class SegmentationVisualizer:
    def __init__(self, class_map, pallete=None, normalized=False, brightness_factor=1.5):
        self.cmap = voc_color_map(N=256, normalized=normalized, brightness_factor=brightness_factor)
        self.class_map = class_map

    def draw(self, image, pred, model_input_shape=None):
        result_images = []
        for _real_gray_image in pred:
            converted_image = self._convert(image, _real_gray_image)
            converted_image = np.transpose(converted_image, (1, 2, 0))
            result_images.append(converted_image)

        return result_images[0]

    def _convert(self, image, gray_image):
        assert len(gray_image.shape) == 2
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = label == gray_image
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = gray_image == 255
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image
