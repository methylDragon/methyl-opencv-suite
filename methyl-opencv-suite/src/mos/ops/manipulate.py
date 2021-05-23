from mos.utils import *
import numpy as np
import cv2

__all__ = [
    'concatenate'
]


################################################################################
# STATIC OPS ===================================================================
################################################################################
def concatenate(*images, fx=0.5, fy=0.5):
    images = list(images)
    img_dim = images[0].shape

    if img_dim[0] >= img_dim[1]:
        axis = 1
    else:
        axis = 0

    for idx, image in enumerate(images):
        if len(image.shape) == 2:
            images[idx] = cv2.cvtColor(images[idx], cv2.COLOR_GRAY2BGR)

    return cv2.resize(
        np.concatenate(images, axis=axis),
        None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC
    )
