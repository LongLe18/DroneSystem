! python3 demo.py --type-model yolov8 \
    --yolo-model weights/detection/v8_small.pt  \
    --reid-model weights/tracking/osnet_x0_25_msmt17.pt  \
    --tracking-method deepocsort \
    --source /media/phuong/DATASET/data_uav/video/bird3.mp4 \
    --conf 0.25  \
    --show \
    --apply-tracking
