import numpy as np
from interpolate import interpolate

# calculate vector e for each update (it is summing values of product
#(I-J)grad(I) on specific region)
def getE (I, J, dx, dy, d, point, window_size):
    e = np.array([[0], [0]])
    for x in xrange(point[0]-window_size[0]/2, point[0]+window_size[0]/2+1):
        for y in xrange(point[1]-window_size[1]/2, point[1]+window_size[1]/2+1):
            diff = I[y, x] - interpolate(J, y+d[1], x+d[0])
            e[0] = e[0]+diff*dx[y, x]
            e[1] = e[1]+diff*dy[y, x]
    return e;
