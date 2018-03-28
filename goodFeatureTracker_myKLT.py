import numpy as np
import cv2
from calculateKLT import calculateKLT

#cap =  cv2.VideoCapture(0) #for tracking object from webcam
cap = cv2.VideoCapture('sailing_boat.mp4')

# Create some random colors
color = np.random.randint(0,255,(100,3))

# params for ShiTomasi corner detection
points = 5
feature_params = dict( maxCorners = points,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#select window size
window_size  = np.array([21, 21])

# points of interest
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    #new frame
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate image alignment
    p1 = calculateKLT(old_gray, frame_gray, p0, window_size, 30, 0.01)

    #get new points to track
    good_new = p1[0:points-1]
    good_old = p0[0:points-1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)

    #if pressed ESC break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
