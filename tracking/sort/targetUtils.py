from filterpy.kalman import KalmanFilter
import numpy as np
from .utilsTracker import iou_score

def convert_bbox_to_features_vector(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = int(bbox[2]) - int(bbox[0])
    h = int(bbox[3]) - int(bbox[1])
    x = int(bbox[0]) + w/2.
    y = int(bbox[1]) + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return [x, y, s, r]

def convert_x_to_bbox(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return [x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]

class targetTrack(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, track):
        """
        Initialises a tracker using initial detection in the format [x1,y1,x2,y2,score,class]
        Params: 
            track: (xmin, ymin) , (xmax, ymax)
        """

        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = np.array(convert_bbox_to_features_vector(track)).reshape(4, 1)
        self.time_since_update = 0
        self.id = targetTrack.count
        targetTrack.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.isSelected = False

    def update(self, track):
        """
        Updates the state vector with observed track.
        """
        self.time_since_update = 0
        # self.history.append(track)
        if (self.hits < 20):
            self.hits += 4
        self.hit_streak += 1
        self.kf.update(np.array(convert_bbox_to_features_vector(track)))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.hits > -10):
            self.hits -= 1
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        a = convert_x_to_bbox(self.kf.x) #.extend([self.score,self.cls])
        self.history.append(a) 
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        
        return convert_x_to_bbox(self.kf.x)
    def get_association_score(self,track):
        iouScr = iou_score(self.history[-1],track[:4])