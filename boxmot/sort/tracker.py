import numpy as np 
from .utilsTracker import iou_batch
from .targetUtils import targetTrack

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
  
def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    if (len(detections) == 0):
        return np.empty((0,2),dtype=int), np.empty((0,5),dtype=int), np.arange(len(trackers))
    
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches) == 0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Tracker(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackList = []
        self.frame_count = 0
        self.selected_track = -1

    def change_selected_track(self):
        if(len(self.trackList) > 0):
            self.selected_track = self.selected_track + 1
        if(self.selected_track > len(self.trackList) - 1):
            self.selected_track = -1
        
        else:
            self.selected_track = -1

    def getSelectedTrack(self):
        if(self.selected_track >= 0):
            if(len(self.trackList) > 0):
                return self.trackList[self.selected_track]
        return None
  
    def update(self, dets=[]):
        """
        Params:
        dets - a list of detections in the format [[x1,y1,x2,y2,score,class],[x1,y1,x2,y2,score,class],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        main_dets = []
        self.frame_count += 1
        # get predicted locations from existing trackers and remove lost tracks
        to_del = []
        for t in range(0, len(self.trackList)):
            # estimate new position of each tracks
            self.trackList[t].predict()
            self.trackList[t].isSelected = (t==self.selected_track)
            # remove tracks that doesn't updated 
            if (self.trackList[t].hits) < -5:
                to_del.append(t)
                if(t == self.selected_track):
                    self.selected_track = -1
        for t in reversed(to_del):
            self.trackList.pop(t)


        trks = np.zeros((len(self.trackList), 4))
        
        # ret = []
        for t, trk in enumerate(trks):
            trk[:] = self.trackList[t].get_state()
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for det in dets: main_dets.append(det.bbox.to_xyxy()) # convert to xyxy box
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(main_dets, trks, self.iou_threshold)
        # update matched trackers with assigned detections
        for m in matched:
        # m[0]: index of object in dets
        # m[1]: indexes of object in trackList
            self.trackList[m[1]].update(main_dets[m[0]])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = targetTrack(main_dets[i]) # (xmin, ymin) , (xmax, ymax)
            self.trackList.append(trk)
        i = len(self.trackList)
        
        return self.trackList