import numpy as np
import math

__all__ = [
    'LinearLine', 'line_angle', 'lines_to_angles'
]

################################################################################
# MODEL INFINITE LINE ==========================================================
################################################################################
class LinearLine:
    """
    Model linear line and support basic operations.

    Supports both polar and rectangular parameters!
    Rect: y = mx + c
    Polar: rho = x * cos(theta) + y * sin(theta)

    Initialise by passing rectangular parameters or 2 or more points.

    Supports resolving x from y and vice versa, as well as finding intercepts
    with bounding boxes.

    You may pass in points denoting line-segments to this to obtain the model
    infinite line!
    """
    def __init__(self, points=None, m=None, c=None):
        assert points is None or (m is None and c is None)

        if points is not None:
            self.points = np.array(points)
            self.m, self.c = LinearLine.linear_rect_params(points)
            self.rho, self.theta = LinearLine.linear_polar_params(points)
        else:
            self.m ,self.c = m, c
            self.points = np.array([
                [0, LinearLine.linear_resolve_y(0, self.m, self.c)],
                [LinearLine.linear_resolve_x(0, self.m, self.c), 0]
            ])
            self.rho, self.theta = LinearLine.linear_polar_params(self.points)

    @property
    def rect_params(self):
        """Returns (m, c) from y = mx + c."""
        return self.m, self.c

    @property
    def polar_params(self):
        """Returns (rho, theta) from rho = x * cos(theta) + y * sin(theta)."""
        return self.rho, self.theta

    def x_from_y(self, y):
        try: # Handle vertical lines
            if math.isinf(self.m):
                return points[0][0]
        except:
            pass
        return LinearLine.linear_resolve_x(y, self.m, self.c)

    def y_from_x(self, x):
        return LinearLine.linear_resolve_y(x, self.m, self.c)

    def intercepts_with_bounding_box(self, bounding_box):
        # start_x <= end_x, start_y <= end_y
        start, end = bounding_box
        m, c = self.m, self.c

        ret = []

        if math.isinf(self.m): # Handle vertical lines
            upper_y_intercept = lower_y_intercept = self.points[0][0]

            if upper_y_intercept <= end[0] and upper_y_intercept >= start[0]:
                ret.append([upper_y_intercept, end[1]])

            if lower_y_intercept <= end[0] and lower_y_intercept >= start[0]:
                ret.append([lower_y_intercept, start[1]])

            return np.array(ret)
        else:
            upper_y_intercept = LinearLine.linear_resolve_x(end[1], m, c)
            lower_y_intercept = LinearLine.linear_resolve_x(start[1], m, c)

        upper_x_intercept = LinearLine.linear_resolve_y(end[0], m, c)
        lower_x_intercept = LinearLine.linear_resolve_y(start[0], m, c)

        # Filter out of bounds
        if upper_y_intercept <= end[0] and upper_y_intercept >= start[0]:
            ret.append([upper_y_intercept, end[1]])

        if lower_y_intercept <= end[0] and lower_y_intercept >= start[0]:
            ret.append([lower_y_intercept, start[1]])

        if upper_x_intercept <= end[1] and upper_x_intercept >= start[1]:
            ret.append([end[0], upper_x_intercept])

        if lower_x_intercept <= end[1] and lower_x_intercept >= start[1]:
            ret.append([start[0], lower_x_intercept])

        return np.array(ret)

    def normal_from_x(self, point, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            normal_m = -1/self.m
        return LinearLine.from_slope((x, self.y_from_x(x)), normal_m)

    def normal_from_y(self, point, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            normal_m = -1/self.m
        return LinearLine.from_slope((self.x_from_y(y), y), normal_m)

    @classmethod
    def from_points(cls, points):
        return cls(points=points)

    @classmethod
    def from_equation(cls, m, c):
        return cls(m=m, c=c)

    @classmethod
    def from_slope(cls, point, m):
        return cls(points=np.array([point,
                                    [point[0] + 1, point[1] + m]]))

    @classmethod
    def from_vector(cls, point, vector):
        return cls(points=np.array(
                [
                    point,
                    [point[0] + vector[0], point[1] + vector[1]]
                ]
            )
        )

    @staticmethod
    def linear_resolve_x(y, m, c): # x = (y - c)/m
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.true_divide(y - c, m)

    @staticmethod
    def linear_resolve_y(x, m, c): # y = mx + c
        with np.errstate(divide='ignore', invalid='ignore'):
            return m * x + c

    @staticmethod
    def linear_slope(points): # Find m from two points
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.true_divide(points[0][1] - points[1][1],
                                  points[0][0] - points[1][0])

    @staticmethod
    def linear_intercept(point, m):
        with np.errstate(divide='ignore', invalid='ignore'):
            return point[1] - m * point[0]

    @staticmethod
    def linear_rect_params(points):
        m = LinearLine.linear_slope(points)
        return (LinearLine.linear_slope(points),
                LinearLine.linear_intercept(points[0], m))

    @staticmethod
    def linear_polar_params(points):
        x_1, y_1 = points[0]
        m = LinearLine.linear_slope(points)

        theta = math.atan2(m, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.true_divide(abs(y_1 - m * x_1),
                                 math.sqrt(m ** 2 + 1))

        if math.isnan(rho): # Handle vertical lines
            rho = x_1

        return rho, theta


################################################################################
# UTILITIES ====================================================================
################################################################################

# MATH =========================================================================
def line_angle(line, degrees=False):
    # Line is: [x_1, y_1, x_2, y_2]. Origin is top left of image.
    radians = math.atan2(line[3] - line[1],
                         line[2] - line[0])

    if degrees:
        return math.degrees(radians) % 360
    else:
        return radians % math.pi

def lines_to_angles(lines, degrees=False, dtype=np.float32):
    if lines.shape != (1, 1, 4):
        lines = np.squeeze(lines)
    else:
        lines = lines.reshape(1, 4)

    angles = map(lambda x:line_angle(x, degrees=degrees), lines)
    return np.fromiter(angles, dtype=dtype).reshape(-1, 1)
