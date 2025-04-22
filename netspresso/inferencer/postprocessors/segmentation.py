import numpy as np


class SegmentationPostprocessor:
    def __init__(self):
        pass

    def __call__(self, outputs, original_shape):
        pred = outputs["pred"]
        H, W = original_shape[-2:]

        # Upsample logits to the images' original size using bilinear interpolation
        pred_upsampled = np.zeros((pred.shape[0], pred.shape[1], H, W))

        for i in range(pred.shape[0]):  # Iterate over the batch
            for j in range(pred.shape[1]):  # Iterate over the channels (classes)
                pred_upsampled[i, j] = self.bilinear_interpolate(pred[i, j], (H, W))

        # Take the argmax over the channel dimension to get the predicted class for each pixel
        pred_classes = np.argmax(pred_upsampled, axis=1)

        return pred_classes

    def bilinear_interpolate(self, img, new_shape):
        H, W = new_shape
        h_old, w_old = img.shape

        # Calculate the ratios
        row_ratio, col_ratio = h_old / H, w_old / W

        # Create a meshgrid of the target image size
        row_coords = np.arange(H) * row_ratio
        col_coords = np.arange(W) * col_ratio

        row_floor = np.floor(row_coords).astype(np.int32)
        col_floor = np.floor(col_coords).astype(np.int32)

        row_ceil = np.ceil(row_coords).clip(max=h_old - 1).astype(np.int32)
        col_ceil = np.ceil(col_coords).clip(max=w_old - 1).astype(np.int32)

        # Get the values of the four surrounding pixels for interpolation
        top_left = img[row_floor[:, None], col_floor]
        top_right = img[row_floor[:, None], col_ceil]
        bottom_left = img[row_ceil[:, None], col_floor]
        bottom_right = img[row_ceil[:, None], col_ceil]

        # Calculate the fractional part of the coordinates
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor

        # Perform bilinear interpolation
        top = top_left * (1 - col_frac) + top_right * col_frac
        bottom = bottom_left * (1 - col_frac) + bottom_right * col_frac
        interpolated = top * (1 - row_frac[:, None]) + bottom * row_frac[:, None]

        return interpolated
