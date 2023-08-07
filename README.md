<div align="center">
<h1>Drone System: Drone detection and Tracking</h1>

<h4>
    A lightweight System for performing surveillance drone 
</h4>
</div>

## <div align="center">Overview</div>

Object detection and object tracking are by far the most important fields of applications in Computer Vision. Specially, drone detection is important problem are still major issues in practical usage. Here we use our proposed models trained on our dataset (mostly drones) and some tracking models (GRM, GOTURN, SEQTRACK) to using these real-world problems

| Folder  | Description  |
|---|---|
| [experiments](https://github.com/LongLe18/DroneSystem/tree/main/experiments)  | experiments of tracking model include GRM, and SEQTrack [GRM](https://github.com/Little-Podi/GRM)/[SEQTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack) model |
| [sahi](https://github.com/LongLe18/DroneSystem/tree/main/sahi)  | A lightweight vision library for performing large scale object detection & instance segmentation [SAHI](https://github.com/obss/sahi) |
| [tracking](https://github.com/LongLe18/DroneSystem/tree/main/tracking)  | source tracking models [GRM](https://github.com/LongLe18/DroneSystem/tree/main/experiments)/[SEQTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack) | 
| [ultralytics](https://github.com/LongLe18/DroneSystem/tree/main/ultralytics)  | Our proposed model based on [YOLOv8](https://github.com/ultralytics/ultralytics) and [SuperYOLO](https://github.com/icey-zhang/SuperYOLO) |

## <div align="center">Quick Start Examples</div>

[📜 Publication that cite our proposed model (currently 40+)]()

## Tutorials

### Requirements

```python
pip install -r requirements.txt
```

### Preparation

- Setup for tracking object

Run the following command to set paths for this project
```
python tracking/init/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
tracking/grm/local.py  # paths about training
tracking/seq/local.py  # paths about testing
```

- Download weight for detection from [google drive]() and tracking object [GRM](https://github.com/Little-Podi/GRM/releases/download/downloads/Trained_Models.zip), [SEQTrack](https://github.com/microsoft/VideoX/blob/master/SeqTrack/MODEL_ZOO.md) models. And path of folder weight (at same root system) is like that

```python
DroneSystem
├── weights
│   ├── detection
│   │   ├── your model v8 or onnx
│   │   ├── yolov8.pt
│   │   ├── yolov8.onnx
│   │   ├── .....
│   ├── tracking
│   │   ├── grm
│   │   │   ├── .....
│   │   ├── seqtrack
│   │   │   ├── .....
```

- Running Demo
```
python demo.py --type-model yolov8 --yolo-model weights/detection/v8_small.onnx --source your_path --conf 0.25  --show --apply-tracking
```
### Optional arguments

Optional arguments:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python demo.py -h` | |  
|  `--type-model` | type model we choose (onnx, yolov8, mmdet,...) | `python demo.py --type-model yolov8` | `onnx` |  
| `--yolo-model` | path to object detection model (onnx, or pt) | `python demo.py --yolo-model weights/detection/v8_small.onnx` | `weights/detection/v8_small.onnx`|  
| `--source` | path to folder contains the list of image files or path to image file | `python demo.py --source /home/whatever/my_detections/` | |  
| `--imgsz` | inference size h,w | `python demo.py --imgsz 640` | `640` |  
| `--conf` | confidence threshold object detection | `python demo.py --conf 0.25` | `0.25` |
| `--iou` | intersection over union (IoU) threshold for NMS | `python demo.py --iou 0.7` | `0.45` | |  
| `--device` | cuda device, i.e. 0 or 0,1,2,3 or cpu |  `python demo.py --device cpu` | `cuda:0` |  
| `--apply-tracking` | apply tracking object | `python demo.py --apply-tracking` | `False` |   
| `--slice ` | apply plugin SAHI to detect small object | `python demo.py --slice` |  `False` |
| `--show` | display tracking video results | `python demo.py --show | `False` |  

## **Acknowledgments**

- Thanks for the great object tracking [GRM](https://github.com/Little-Podi/GRM),
[SEQTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack),
[GOTURN](https://github.com/davheld/GOTURN).
- For object detection, we use [YOLOv8](https://github.com/ultralytics/ultralytics),
[SuperYOLO](https://github.com/icey-zhang/SuperYOLO).