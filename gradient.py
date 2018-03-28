import numpy as np
from scipy import signal

def conv2(x, y, mode='same'):
    return np.rot90(signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

# calculates derivative in x and y direction
def gradient(I):

    #use scharr to get gradient
	scharr = np.array([[ 3, 0, -3], [10, 0, -10 ],[ 3, 0, -3]])/32.
	dx = conv2(I, scharr)
	dy=  conv2(I, np.transpose(scharr))

	return dx, dy
