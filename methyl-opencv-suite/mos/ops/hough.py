from mos.utils import *
import mos.utils

import numpy as np
import cv2

__all__ = [
    'draw_hough_lines',
    'draw_hough_circles',
    'hough_lines_probabilistic',
    'hough_circles',
    'get_mask_from_parallel_lines'
]


################################################################################
# UTILITIES ====================================================================
################################################################################

# FILTERING ====================================================================
def get_mask_from_parallel_lines(image,
                                 lines,
                                 bandwidth=np.pi/4,
                                 bin_seeding=True):
    hull_img = np.zeros(image.shape, np.uint8)

    # Find equivalence classes of parallel lines
    angles = mos.utils.lines_to_angles(lines)
    line_args = mos.utils.meanshift_cluster(angles,
                                            bandwidth=bandwidth,
                                            bin_seeding=bin_seeding,
                                            as_arg=True)

    height, width = image.shape[:2]
    image_box = ((0, 0), (width, height))

    # Compute convex hull and fill polygon for each class
    for cluster_args in line_args:
        cluster_lines = lines[cluster_args] # Get lines by cluster args
        intersects = []

        for line in cluster_lines:
            line = line.flatten()
            l_line = mos.utils.LinearLine(((line[0], line[1]),
                                           (line[2], line[3])))
            intersects.append(
                l_line.intercepts_with_bounding_box(((0, 0), (width, height)))
            )

        try: # Very rarely, this reshape fails.
            intersects_arr = (
                np.array(intersects).reshape(-1, 2).astype(np.int)
            )
        except:
            print("INTERSECT EXCEPTION")
            continue

        convex_hull_pts = cv2.convexHull(intersects_arr)

        # Draw all contours
        return cv2.fillConvexPoly(hull_img, convex_hull_pts, (255, 255, 255))

# DISPLAY ======================================================================
def draw_hough_lines(img, lines, color=(0, 255, 0), thickness=2):
    if lines is not None:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)

    return img

def draw_hough_circles(img, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            cv2.circle(img, (i[0],i[1]), i[2], (0,255,0), 2) # Circumference
            cv2.circle(img, (i[0],i[1]), 2, (0,0,255), 3) # Center

    return img


################################################################################
# DYNAMIC OPS ==================================================================
################################################################################
@TrackbarManager(
    "Hough Lines Probabilistic Params",
    {
        "rho": {'default': 1, 'max': 20},
        "theta": {'default': 12,
                  'max': 50,
                  'operation': lambda x: x * np.pi / 1800},
        "threshold": {'default': 122, 'max': 1000},
        "min_line_length": {'default': 62, 'max': 100},
        "max_line_gap": {'default': 160, 'max': 1000}
     }
)
def hough_lines_probabilistic(image,
                              rho,
                              theta,
                              threshold,
                              min_line_length,
                              max_line_gap):
    assert type(image) is np.ndarray, "Input image must be OpenCV image!"
    return cv2.HoughLinesP(image,
                           rho=clamp(rho, 1),
                           theta=clamp(theta, 1 * np.pi / 1800),
                           threshold=threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)

@TrackbarManager(
    "Hough Circles Params",
    {
        "inverse_resolution": {'default': 1,
                               'max': 100,
                               'operation': lambda x: x * 0.1},
        "min_distance": {'default': 32, 'max': 255},
        "canny_upper": {'default': 200, 'max': 255},
        "threshold": {'default': 40, 'max': 255},
        "min_radius": {'default': 0, 'max': 255},
        "max_radius": {'default': 2, 'max': 255}
     }
)
def hough_circles(image,
                  inverse_resolution,
                  min_distance,
                  canny_upper,
                  threshold,
                  min_radius,
                  max_radius):
    assert type(image) is np.ndarray, "Input image must be OpenCV image!"
    return cv2.HoughCircles(image,
                            method=cv2.HOUGH_GRADIENT,
                            dp=clamp(inverse_resolution, 1),
                            minDist=clamp(min_distance, 1),
                            param1=clamp(canny_upper, 1),
                            param2=clamp(threshold, 1),
                            minRadius=min_radius,
                            maxRadius=max_radius)
