from functools import wraps, update_wrapper
import numpy as np
import cv2


################################################################################
# UTILITIES ====================================================================
################################################################################
MORPH_SHAPES = (cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE)

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

    def start(self):
        if self._started:
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
                 use_trackbar_params=False,
                 **kwargs):
        if use_trackbar_params:
            return self.from_trackbars(*args, **kwargs)
        elif get_params:
            return {self._func.__name__ : self.get_params()}
        return self._func(*args, **kwargs)

    def __call__(self, f):
        self._func = f
        update_wrapper(self._wrapper.__func__, f)
        return self._wrapper

    def __get__(self, instance):
        return types.MethodType(self, instance) if instance else self
