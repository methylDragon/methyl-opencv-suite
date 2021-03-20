from functools import wraps
import numpy as np
import cv2


################################################################################
# UTILITIES ====================================================================
################################################################################
MORPH_SHAPES = (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS)

# BASIC ========================================================================
def clamp(n, smallest=float('-inf'), largest=float('inf')):
    return max(smallest, min(n, largest))

def even(n, ceiling=False): # Force even
    return (n if not n % 2 else n - 1 + ceiling * 2)

def odd(n, ceiling=False): # Force odd
    return (n if n % 2 else n - 1 + ceiling * 2)

def map_to_int(*args):
    return lambda x: args[x]

# HOUGH ========================================================================
def draw_hough_lines(img, lines):
    if lines is not None:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)

    return img
