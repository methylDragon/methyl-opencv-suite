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


################################################################################
# TRACKBAR DECORATOR ===========================================================
################################################################################
def opencv_trackbars(window_name,
                     param_definitions,
                     callback=lambda *args: None):
    """
    Decorator to manage OpenCV trackbars and pass args to wrapped function.

    The wrapped function will be able to generate OpenCV trackbars by passing
    start_trackbars = True.

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
    For each wrapped function func, to use trackbars, first call:
    func(start_trackbars=True)

    Then call this to use parameter values from the trackbars:
    func(use_trackbar_params=True)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args,
                    start_trackbars=False,
                    use_trackbar_params=False,
                    **kwargs):
            if start_trackbars:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                for param_var_name, definitions in param_definitions.items():
                    cv2.createTrackbar(
                        definitions.get('name', param_var_name),
                        window_name,
                        definitions.get('default', 0),
                        definitions.get('max', 100),
                        callback
                    )
                return

            if use_trackbar_params:
                func_params = {}

                for param_var_name, definitions in param_definitions.items():
                    op = definitions.get('operation', lambda x: x)
                    min = definitions.get('min', float('-inf'))

                    func_params[param_var_name] = clamp(op(
                        cv2.getTrackbarPos(
                            definitions.get('name', param_var_name),
                            window_name
                        )
                    ), min)

                return func(*args, **kwargs, **func_params)
            return func(*args, **kwargs)
        return wrapper
    return decorator
