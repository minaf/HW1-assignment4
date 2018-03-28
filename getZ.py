import numpy as np

# calculate matrix Z for each update (it is summing values of product
#grad(I)grad(I)' on specific region)
#dx and dy are gradient direction of image I
def getZ(dx, dy, point, window_size):
    Z = np.zeros([2, 2])
    for x in xrange(point[0]-(window_size[0]-1)/2, point[0]+(window_size[0]-1)/2+1):
        for y in xrange(point[1]-(window_size[1]-1)/2, point[1]+(window_size[1]-1)/2+1):
            Z[0, 0] = Z[0, 0]+dx[y, x]*dx[y, x]
            Z[0, 1] = Z[0, 1]+dx[y, x]*dy[y, x]
            Z[1, 0] = Z[1, 0]+dy[y, x]*dx[y, x]
            Z[1, 1] = Z[1, 1]+dy[y, x]*dy[y, x]
    return Z
