from mos.utils import *
import numpy as np
import cv2

__all__ = [
    'canny'
]


################################################################################
# OPERATIONS ===================================================================
################################################################################
@TrackbarManager(
    "Canny Params",
    {
        "blur_amount": {'default': 25, 'max': 100},
        "min_thresh": {'default': 4344, 'max': 10000},
        "max_thresh": {'default': 2511, 'max': 10000},
        "aperture": {'default': 7, 'max': 7},
        "l2_grad": {'default': 1, 'max': 1},
     }
)
def canny(image_in, blur_amount=13, min_thresh=2900, max_thresh=4240, aperture=7, l2_grad=1):
    assert type(image_in) is np.ndarray, "Input image must be OpenCV image!"
    blur_amount = odd(clamp(blur_amount, 1))
    aperture = odd(clamp(aperture, 3, 7))

    return cv2.Canny(
        cv2.GaussianBlur(image_in,(blur_amount,blur_amount),0),
        min_thresh,
        max_thresh,
        None,
        aperture,
        l2_grad
    )
