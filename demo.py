import torch
import argparse
from pathlib import Path
from sahi.predict import predict_new
from tracking.utils import logger as LOGGER
from tracking.utils.torch_utils import select_device


@torch.no_grad()
def run(args):
    select_device(args['device'])
    LOGGER.info(args)

    predict_new(
        model_type=args['type_model'],
        model_path=args['yolo_model'],
        model_device=args['device'],
        model_confidence_threshold=args['conf'],
        source=args['source'],
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        view_video=args['show'],
        no_sliced_prediction=args['slice'],
        apply_tracking=args['apply_tracking'],
        image_size=args['imgsz'],
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type-model', type=str, default='onnx', help='type model: onnx, yolov8, mmdet, ...')
    parser.add_argument('--yolo-model', type=Path, default='weights/detection/v8_small.onnx', help='model.onnx path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold object detection')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--apply-tracking', action='store_true', help='apply tracking object')
    parser.add_argument('--slice', action='store_true', help='apply plugin SAHI to detect small object')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--hide-label', action='store_true', help='hide labels when show')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true', help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true', help='save tracking results in a single txt file')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
