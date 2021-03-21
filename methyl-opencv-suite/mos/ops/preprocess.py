from mos.utils import *
import numpy as np
import cv2

__all__ = [
        'preprocess_clahe_blur', 'blur', 'threshold', 'morph_noise_remove'
]


################################################################################
# STATIC OPS ===================================================================
################################################################################
def preprocess_clahe_blur(image_in):
    assert type(image_in) is np.ndarray, "Input image must be OpenCV image!"

    # Adaptive Histogram Equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    lab = cv2.cvtColor(image_in, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)
    image_in = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gaussian Blur
    # image_out = cv2.GaussianBlur(image_in,(5,5),0)

    return image_in


################################################################################
# DYNAMIC OPS ==================================================================
################################################################################
@TrackbarManager(
    "Blur Params",
    {
        "x_kernel": {'max': 100, 'operation': lambda x: clamp(x, 1)},
        "y_kernel": {'max': 100, 'operation': lambda x: clamp(x, 1)},
        "sigma_x": {'max': 100}
     }
)
def blur(image, x_kernel, y_kernel, sigma_x):
    assert type(image) is np.ndarray, "Input image must be OpenCV image!"
    return cv2.GaussianBlur(image,(odd(x_kernel),odd(y_kernel)),sigma_x)

    return image_out

@TrackbarManager(
    "Threshold Params",
    {
        "max_thresh": {'default': 255, 'max': 255},
        "neighbour_sz": {'min': 1, 'default': 32, 'max': 100},
        "subtraction": {'default': 2, 'max': 50}
     }
)
def threshold(image_in, max_thresh=255, neighbour_sz=16, subtraction=2):
    assert type(image_in) is np.ndarray, "Input image must be OpenCV image!"
    return cv2.adaptiveThreshold(
        cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY),
        max_thresh,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        odd(clamp(neighbour_sz, 3)),
        subtraction
    )

@TrackbarManager(
    "Morph Noise Remove Params",
    {
        "close_kernel": {'default': 1, 'max': 2,
                         'operation': map_to_int(*MORPH_SHAPES)},
        "close_kernel_size": {'default': 5, 'max': 10},
        "open_kernel": {'default': 1, 'max': 2,
                         'operation': map_to_int(*MORPH_SHAPES)},
        "open_kernel_size": {'default': 6, 'max': 10},
        "erode_kernel": {'default': 1, 'max': 2,
                         'operation': map_to_int(*MORPH_SHAPES)},
        "erode_kernel_size": {'default': 6, 'max': 10},
        "erode_iter": {'default': 1, 'max': 10},
     }
)
def morph_noise_remove(image_in,
                       close_kernel=cv2.MORPH_CROSS, open_kernel=cv2.MORPH_CROSS,
                       close_kernel_size=5, open_kernel_size=6,
                       erode_kernel=cv2.MORPH_CROSS, erode_kernel_size=6,
                       erode_iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                       (close_kernel_size, close_kernel_size))
    ret = cv2.morphologyEx(image_in, cv2.MORPH_CLOSE, kernel)
    ret = cv2.bitwise_not(ret)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                       (open_kernel_size, open_kernel_size))
    ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                       (erode_kernel_size, erode_kernel_size))
    ret = cv2.erode(ret, kernel, iterations=erode_iter)
    return ret
