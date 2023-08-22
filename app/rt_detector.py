import time
from tracking.utils import logger as LOGGER
from sahi.auto_model import AutoDetectionModel
from tracking.utils.torch_utils import select_device

LOW_MODEL_CONFIDENCE = 0.1

MODEL_TYPE_TO_WEIGHT = {
    "yolov8": "../weights/detection/1280v8m.pt",
    "onnx": "../weights/detection/1280v8m.onnx",
}

class Detector:
    def __init__(self, model_type: str = 'onnx', 
                 model_confidence_threshold: float = 0.25,
                 model_device: str = 'cuda:0',
                 image_size: int = 1280):
        
        self.model_type = model_type
        self.model_confidence_threshold = model_confidence_threshold
        self.model_device = model_device
        self.image_size = image_size

        self.detection_model = None
        select_device(self.model_device)
        self.init_model()

    def init_model(self):
        if self.detection_model is None:
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=self.model_type,
                model_path=MODEL_TYPE_TO_WEIGHT[self.model_type],
                config_path=None,
                confidence_threshold=self.model_confidence_threshold,
                device=self.model_device,
                category_mapping=None,
                category_remapping=None,
                load_at_init=False,
                image_size=self.image_size,
            )
            self.detection_model.load_model()
            
        config = {'type_model': self.model_type, 
                  'yolo_model': MODEL_TYPE_TO_WEIGHT[self.model_type], 
                  'imgsz': self.image_size, 
                  'conf': self.model_confidence_threshold, 
                  'iou': 0.7, 
                  'device': self.model_device, 'apply_tracking': True, 
                }
        LOGGER.info(config)
