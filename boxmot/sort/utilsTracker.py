import numpy as np

def iou_batch(bb_test, bb_gt):                                                                                                   
    """                                                                                                                      
    From SORT: Computes IUO between two bboxes in the form [l,t,w,h]   
    return (area of iou)/(area1+area2)
    """                                                                                                                      
    bb_gt = np.expand_dims(bb_gt, 0)                                                                                         
    bb_test = np.expand_dims(bb_test, 1)   
    
    # outer bb
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])                                                                         
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])                                                                         
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])                                                                         
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3]) 
    #outer w h                                                                        
    w = np.maximum(0., xx2 - xx1)                                                                                            
    h = np.maximum(0., yy2 - yy1)            
    #outer area                                                                             
    wh = w * h                                                                                                               
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)  

def iou_score(bb_test, bb_gt):                                                                                                   
    """                                                                                                                      
    From SORT: Computes IUO between two bboxes in the form [l,t,r,b]   
    return (area of iou)*2/(area1+area2)
    """                                                                                                                      

    # iou bb
    xx1 = np.maximum(bb_test[0], bb_gt[0])                                                                         
    yy1 = np.maximum(bb_test[1], bb_gt[1])                                                                         
    xx2 = np.minimum(bb_test[2], bb_gt[2])                                                                         
    yy2 = np.minimum(bb_test[3], bb_gt[3]) 
    #iou w h                                                                        
    w = np.maximum(0., xx2 - xx1)                                                                                            
    h = np.maximum(0., yy2 - yy1)            
    #iou area                                                                             
    wh = w * h   
    #   iou score                                                                                                           
    o = wh*2 / ((bb_test[2] - bb_test[ 0]) * (bb_test[ 3] - bb_test[1])                                      
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[ 3] - bb_gt[1]) - wh)
    return(o) 