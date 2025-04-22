import numpy as np


def voc_color_map(N=256, normalized=False, brightness_factor=1.5):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3

        # Adjust brightness
        r = min(255, max(0, r * brightness_factor))
        g = min(255, max(0, g * brightness_factor))
        b = min(255, max(0, b * brightness_factor))

        cmap[i] = np.array([r, g, b])

    if normalized:
        cmap = cmap / 255
    return cmap
