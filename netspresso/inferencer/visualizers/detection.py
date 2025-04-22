import cv2
import numpy as np
from matplotlib import pyplot as plt

from netspresso.inferencer.visualizers.utils import voc_color_map


class DetectionVisualizer:
    def __init__(self, class_map=None, normalized=False, brightness_factor=1.5):
        self.cmap = voc_color_map(N=256, normalized=normalized, brightness_factor=brightness_factor)
        self.class_map = class_map

    def draw(self, image, pred, model_input_shape=None, text_scale=0.7, text_thickness=2, bbox_thickness=2):
        resize_factor = max((image.shape[0] / model_input_shape[0]), (image.shape[1] / model_input_shape[1]))
        pred[0][:, 1::2] *= resize_factor
        pred[0][:, :4:2] *= resize_factor

        visualize_image = image.copy()
        for bbox_label, class_label in zip(pred[0], pred[1]):
            # Get bbox coordinates
            x1 = int(bbox_label[0])
            y1 = int(bbox_label[1])
            x2 = int(bbox_label[2])
            y2 = int(bbox_label[3])

            # Get bbox color
            color = self.cmap[class_label].tolist()

            # Draw bbox
            visualize_image = cv2.rectangle(visualize_image, (x1, y1), (x2, y2), color=color, thickness=bbox_thickness)

            # Get class name
            class_name = self.class_map[int(class_label)] if self.class_map else str(class_label)

            # Draw class info
            text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
            text_w, text_h = text_size
            visualize_image = cv2.rectangle(
                visualize_image, (x1, y1 - 5 - text_h), (x1 + text_w, y1), color=color, thickness=-1
            )
            visualize_image = cv2.putText(
                visualize_image,
                str(class_name),
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
            )

        return visualize_image

    def visualize_by_plt(self, image):
        plt.imshow(image)
        plt.show()
