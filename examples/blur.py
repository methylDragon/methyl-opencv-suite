'''
  methyl-opencv-suite (MOS) Examples

  Blur example with dynamic trackbar params.


                                   .     .
                                .  |\-^-/|  .
                               /| } O.=.O { |\
                              /´ \ \_ ~ _/ / `\
                            /´ |  \-/ ~ \-/  | `\
                            |   |  /\\ //\  |   |
                             \|\|\/-""-""-\/|/|/
                                     ______/ /
                                     '------'
                       _   _        _  ___
             _ __  ___| |_| |_ _  _| ||   \ _ _ __ _ __ _ ___ _ _
            | '  \/ -_)  _| ' \ || | || |) | '_/ _` / _` / _ \ ' \
            |_|_|_\___|\__|_||_\_, |_||___/|_| \__,_\__, \___/_||_|
                               |__/                 |___/
            -------------------------------------------------------
                            github.com/methylDragon

  Try it out! Press 't' to toggle between directly passing parameters and
  using parameters from the trackbar.

  Press 'q' to quit. Upon quitting, you should see a printout of the trackbar
  parameters.

  ---

  License: Apache 2.0

'''

import mos.ops

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, image = cap.read()

static_params = False

while True:
    for i in range(3):
        ret, image = cap.read()

    if ret:
        if static_params:
            blurred = mos.ops.blur(image, 100, 100, 0)
        else:
            blurred = mos.ops.blur(image, use_trackbar_params=True)
        cv2.imshow("Blur Params", blurred)

    key = cv2.waitKey(100)

    if key == ord('q'):
        break
    elif key == ord('t'):
        static_params ^= 1

print(mos.ops.blur(get_params=True))

cv2.destroyAllWindows()
cap.release()
