# OBSS SAHI Tool
# Code written by Karl-Joan Alesma and Michael García, 2023.

import logging
from typing import Any, List, Optional
import ast
import numpy as np
import torch

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
from ultralytics.yolo.utils import LOGGER, ops
from ultralytics.yolo.engine.results import Results, LetterBox


class ONNXDetectionModel(DetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max supression, defaults to 0.7.
        """
        super().__init__(*args, **kwargs)
        self.metadata = None
        self.output_names = []
        self.iou_threshold = iou_threshold

    def load_model(self):
        """Detection model is initialized and set to self.model.
        Options for onnxruntime sessions can be passed as keyword arguments.
        """
        import onnxruntime
        try:
            w = str(self.model_path[0] if isinstance(self.model_path, list) else self.model_path)
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and self.device != 'cpu'  # use CUDA
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            self.output_names = [x.name for x in session.get_outputs()]
            self.metadata = session.get_modelmeta().custom_metadata_map  # metadata
            self.set_model(session)
        except Exception as e:
            raise TypeError("model_path is not a valid onnx model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying ONNX model.
        Args:
            model: Any
                A ONNX model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(key): value for key, value in ast.literal_eval(self.category_names).items()}
            self.category_mapping = category_mapping
    
    def _preprocess_image(self, image):
        """Prepapre image for inference by resizing, normalizing and changing dimensions.
        Args:
            image: np.ndarray
                Input image with color channel order BGR.
        """ 
        if not isinstance(image, torch.Tensor):
            image = np.stack([LetterBox([self.image_size, self.image_size], auto=False, stride=32)(image=x) for x in image])
            image = image[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            image = np.ascontiguousarray(image)  # contiguous
            image = torch.from_numpy(image)

        # NOTE: assuming im with (b, 3, h, w) if it's a tensor
        img = image.to(self.device)
        fp16 = False
        img = img.half() if fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def _post_process(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.confidence_threshold,
                                        self.iou_threshold,
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img=orig_img, path=None, names=self.category_mapping, boxes=pred))
        return results

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
    
        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        im = self._preprocess_image([image[:, :, ::-1]])
        im = im.cpu().numpy()  # torch to numpy
        y = self.model.run(self.output_names, {self.model.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            outputs = self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            outputs = self.from_numpy(y)
        prediction_result = self._post_process(outputs, im, [image[:, :, ::-1]])

        prediction_result = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold] for result in prediction_result
        ]
        self._original_predictions = prediction_result

    @property
    def category_names(self):
        return self.metadata['names']
    
    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    def from_numpy(self, x):
        """
         Convert a numpy array to a tensor.

         Args:
             x (np.ndarray): The array to be converted.

         Returns:
             (torch.Tensor): The converted tensor
         """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x
    