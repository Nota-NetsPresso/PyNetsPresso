import cv2


class ClassificationVisualizer:
    def __init__(self, class_map, pallete=None) -> None:
        self.n = len(class_map)
        self.class_map = class_map

    def draw(self, image, pred, model_input_shape=None, text_scale=0.7, text_thickness=2):
        visualize_image = image.copy()

        class_name = self.class_map[int(pred[0])]  # Class is determined with top1 score

        color = (0, 0, 255)
        x1, y1 = 0, 0
        text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_w, text_h = text_size
        visualize_image = cv2.rectangle(
            visualize_image, (x1, y1), (x1 + text_w, y1 + text_h + 5), color=color, thickness=-1
        )
        visualize_image = cv2.putText(
            visualize_image,
            str(class_name),
            (x1, y1 + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
        )

        return visualize_image
