import numpy as np
import cv2
from calculateKLT import calculateKLT

#cap =  cv2.VideoCapture(0) #for tracking object from webcam
cap = cv2.VideoCapture('sailing_boat.mp4')

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#select ROI manually
r = cv2.selectROI(old_gray)
#select window size
window_size  = np.array([21, 21])

#define params for function to calculate optical flow
lk_params = dict( winSize  = (window_size[0], window_size[1]),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# points of interest
p0 = np.empty([4, 1, 2], dtype=np.float32)
p0[0][0] = [r[0],r[1]]
p0[1][0] = [r[0]+r[2],r[1]]
p0[2][0] = [r[0],r[1]+r[3]]
p0[3][0] = [r[0]+r[2],r[1]+r[3]]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    #get new frame
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[0:4]
    good_old = p0[0:4]

    #drawing rectangle of tracked object
    cv2.rectangle(frame,(p1[0][0, 0], p1[0][0, 1]),(p1[3][0, 0], p1[3][0, 1]),(0,255,0),3)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)

    #if press ESC break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
