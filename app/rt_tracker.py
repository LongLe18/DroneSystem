from tracking.utils import logger as LOGGER
from tracking.sort.tracker import Tracker as TrackerSORT
from tracking.grm.mainTracker import Tracker as GRMTracker
from tracking.seq.mainTracker import Tracker as SEQTracker


MODEL_TYPE_TO_WEIGHT = {
    "grm": "vitb_256_ep300",
    "seqtrack": "seqtrack_b256",
}

class Tracker:
    def __init__(self, model_type: str = 'grm'):
        
        self.model_type = model_type
        self.params = None
        self.root_s_tracker = None
        self.tracker = None

    def load_parameter(self):
        params = self.root_s_tracker.get_parameters()
        params.debug = 0
        params.tracker_name = self.root_s_tracker.name
        params.param_name = self.root_s_tracker.parameter_name
        return params
    
    def init_model(self):
        if self.model_type == 'grm':
            self.root_s_tracker = GRMTracker(self.model_type, MODEL_TYPE_TO_WEIGHT[self.model_type], None, None)
        elif self.model_type == 'seqtrack':
            self.root_s_tracker = SEQTracker(self.model_type, MODEL_TYPE_TO_WEIGHT[self.model_type], None, None)

        self.params = self.load_parameter()
        self.tracker = TrackerSORT()

        config = {
            'model_type': self.model_type,
            'backbond': MODEL_TYPE_TO_WEIGHT[self.model_type],
            'params': self.params,
        }
        LOGGER.info(config)
    
    def create_tracker(self):
        return self.root_s_tracker.create_tracker(self.params)
