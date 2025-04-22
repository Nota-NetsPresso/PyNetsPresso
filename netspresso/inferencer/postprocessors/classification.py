from typing import Optional

import numpy as np


class ClassificationPostprocessor:
    def __init__(self, top_k_max=20):
        self.top_k_max = top_k_max

    def __call__(self, outputs, k: Optional[int] = None):
        pred = outputs["pred"]

        maxk = min(self.top_k_max, pred.shape[1])
        if k:
            maxk = min(k, maxk)

        # Find the indices of the top k values along the second axis (classes)
        pred_indices = np.argsort(-pred, axis=1)[:, :maxk]

        return pred_indices
