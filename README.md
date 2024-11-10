This code is used for tracking the region of interest (ROI) in the video

Code
testKLT.py - testing image alignment for view0.png and view1.png
myROI_tracker.py - track ROI(manually selected) using implemented algorithm
cvROI - track ROI(manually selected) using Open CV function calcOpticalFlowPyrLK
goodFeatureTracker_myKLT.py - track points (selected with Open CV function goodFeaturesToTrack) using implemented algorithm
cvROI - track points (selected with Open CV function goodFeaturesToTrack) using Open CV function calcOpticalFlowPyrLK

Video
swan.mov - used for checking the effect of rotation
fish.mp4 - used for checking the effect of light change
