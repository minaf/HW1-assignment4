import numpy as np
import cv2

from calculateKLT import calculateKLT

# load images
I = cv2.imread('view0.png',0)
J = cv2.imread('view1.png',0)

# define some points to track
p0 = np.empty([2, 1, 2], dtype=np.float32)
p0[0][0] = [320, 336]    # points of interest
p0[1][0] = [336, 320]    # points of interest


# parameters for tracking algorithm
window_size = np.array([21,21])          # window size
maxIter = 30           # maximum iterations
minDisp = 0.01         # minimum disparity threshold

# calculate point match
p1 = calculateKLT(I, J, p0, window_size, maxIter, minDisp)
# Test implementation using openCV (single scale tracking - maxLevel = 0)
lk_params = dict( winSize  = (window_size[0], window_size[1]),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, maxIter, minDisp))
p1ocv = cv2.calcOpticalFlowPyrLK(I, J, p0, None, **lk_params)

# print results
print(p1)
print(p1ocv)
