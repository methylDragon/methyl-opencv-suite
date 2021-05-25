from functools import wraps, update_wrapper
import numpy as np
import cv2

__all__ = [
    'MORPH_SHAPES', 'clamp', 'even', 'odd', 'map_to_int', 'TrackbarManager',
    'midpoint_from_dimensions', 'midpoint_from_points', 'circle_to_box',
    'circles_to_boxes', 'draw_boxes'
]

################################################################################
# UTILITIES ====================================================================
################################################################################
MORPH_SHAPES = (cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE)

# BASIC ========================================================================
def map_to_int(*args):
    return lambda x: args[x]

# BASIC MATH ===================================================================
def clamp(n, smallest=float('-inf'), largest=float('inf')):
    return max(smallest, min(n, largest))

def even(n, ceiling=False): # Force even
    return (n if not n % 2 else n - 1 + ceiling * 2)

def odd(n, ceiling=False): # Force odd
    return (n if n % 2 else n - 1 + ceiling * 2)

# BASIC GEOMETRY ===============================================================
def midpoint_from_dimensions(dimensions):
    return dimensions / 2

def midpoint_from_points(points):
    if type(points) is not np.ndarray:
        points = np.array(points)
    assert points.dtype is not np.dtype(object), "Input must be regular!"

    point = []
    for i in range(len(points)):
        point.append(np.mean(points[:,i]))
    return np.array(point).astype(points.dtype)

def circle_to_box(circle):
    x, y, r = circle
    return np.array([[x-r, y-r], [x+r, y+r]])

def circles_to_boxes(circles):
    if circles is not None:
        circles = np.around(circles)
        boxes = []

        for circle in circles[0,:]:
            boxes.append(circle_to_box(circle))
        return np.array(boxes).astype(np.int16)
    else:
        return np.array([]).astype(np.int16)

def draw_boxes(img, boxes, color=(0,255,0), thickness=2):
    if boxes is not None:
        ret = np.copy(img)

        for box in boxes:
            x_1, y_1 = box[0]
            x_2, y_2 = box[1]

            # This inplace draws
            cv2.rectangle(ret, (x_1, y_1), (x_2, y_2), color, thickness)

        return ret
    else:
        return img

################################################################################
# TRACKBAR DECORATOR ===========================================================
################################################################################
class TrackbarManager(object):
    """
    Decorator to manage OpenCV trackbars and pass args to wrapped function.

    The wrapped function will be able to generate OpenCV trackbars and read
    off of them the moment it is called with use_trackbar_params=True.

    The decorator call must be passed definitions for each parameter:
    {
        'param_var_name': {'name': str=PARAM_VAR_NAME,
                           'default': int=0,
                           'min': int=float('-inf'),
                           'max': int=100,
                           'operation': function=lambda x: x}
    }

    Parameter Definitions
    ---------------------
    param_var_name : str
        The variable name that the wrapped function will use.
    name : str, optional
        Trackbar name.
    min : int, optional
        Min value of trackbar.
    default : int, optional
        Starting value of trackbar.
    max : int, optional
        Max value of trackbar.
    operation : function, optional
        Operation to run on trackbar output before passing to wrapped function.

    Note
    ----
    Pass use_trackbar_params = True to use parameter values from the trackbars.
    (So, func(*args, use_trackbar_params=True))

    If it is the first time you are calling it, trackbar windows will
    automatically initialise.

    If you want to get the parameters values from the trackbars, pass
    get_params = True.
    (So, func(*args, get_params=True))

    If you want to force-start the trackbars start_trackbars = True.
    (So, func(*args, start_trackbars=True))
    """
    def __init__(self,
                 window_name,
                 param_definitions,
                 callback=lambda *args: None):
        self._window_name = window_name
        self._param_definitions = param_definitions
        self._callback = callback

        self._started = False
        self._func = None

    def start(self, force=False):
        if self._started and not force:
            return

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        for param_var_name, definitions in self._param_definitions.items():
            cv2.createTrackbar(
                definitions.get('name', param_var_name),
                self._window_name,
                definitions.get('default', 0),
                definitions.get('max', 100),
                self._callback
            )

        self._started = True
        return

    def from_trackbars(self, *args, **kwargs):
        return self._func(*args, **kwargs, **self.get_params())

    def get_params(self):
        if not self._started:
            self.start()

        if cv2.getWindowProperty(self._window_name, 3) < 0:
            self.start(force=True)

        func_params = {}

        for param_var_name, definitions in self._param_definitions.items():
            op = definitions.get('operation', lambda x: x)
            min = definitions.get('min', float('-inf'))

            func_params[param_var_name] = clamp(op(
                cv2.getTrackbarPos(definitions.get('name', param_var_name),
                                   self._window_name)
            ), min)

        return func_params

    def _wrapper(self,
                 *args,
                 get_params=False,
                 start_trackbars=False,
                 use_trackbar_params=False,
                 **kwargs):
        if use_trackbar_params:
            return self.from_trackbars(*args, **kwargs)
        elif start_trackbars:
            self.start(force=True)
            return
        elif get_params:
            return {self._func.__name__ : self.get_params()}
        return self._func(*args, **kwargs)

    def __call__(self, f):
        self._func = f
        update_wrapper(self._wrapper.__func__, f)
        return self._wrapper

    def __get__(self, instance):
        return types.MethodType(self, instance) if instance else self
