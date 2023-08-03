import time
import cv2
import numpy as np
import math

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

class Single_tracker(object):
    def __init__(self, tracker_type, frame, initBB):
        self.tracker_type = tracker_type
        self.tracker = cv2.TrackerMOSSE_create()
        self.tracker.init(frame, initBB)
        self.track_bbox = initBB
        self.success = True
        self.h_path = []
        self.h_path.append(self.track_bbox)


    def update(self, frame):
        (self.success, self.track_bbox) = self.tracker.update(frame)
        self.h_path.append(self.track_bbox)

        return (self.success, self.track_bbox)

    def get_tracker_status(self):
        return self.success

    def get_track_bbox(self):
        return self.track_bbox

