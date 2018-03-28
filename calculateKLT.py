import numpy as np
from gradient import gradient
from getZ import getZ
from getE import getE

# calculate image alignment for I and J
# p0 - list of points to match
# window_size - window size around points of interest
# maxIter - maximum correcting updates for total displacement
# minDisp - stoping criteria treshold |d|
def calculateKLT(I, J, p0, window_size, maxIter, stopCriteria):
    # the number of points for tracking
    n = len(p0)

    # calculate the gradient of image I
    dx, dy = gradient(I)

    # get d for each point
    p1 =  np.zeros_like(p0)

    for i in range(n):
        point = np.array([int(p0[i][0][0]), int(p0[i][0][1])])
        # get matrix Z
        Z = getZ(dx, dy, point, window_size)
        # iterative algorithm to obtain value of d
        dtot = np.array([0, 0]) #initialize dtot for each point
        for ii in range(maxIter):

            # get vector e
            e = getE(I, J, dx, dy, dtot, point, window_size)
            # get displacement correction
            ds = np.linalg.solve(Z, e)
            d =  np.array([ds[0][0], ds[1][0]])

            # update total displacement
            dtot = dtot + d

            # check termination criteria
            if np.linalg.norm(d) < stopCriteria:
                break

        # save result
        p1[i][0] = p0[i][0] + dtot


    return p1
