! python demo.py --type-model onnx \
    --yolo-model weights/detection/v8_small.onnx  \
    --source D:\\PracticePY\\Project\\Drone-2023\\data\\video1.avi \
    --conf 0.25  \
    --show \
    --apply-tracking

# demo sequence tracking
# python3 demoSeq.py seqtrack seqtrack_b256 --video_path /media/phuong/DATASET/data_uav/video/video6.mp4

# demo GRM tracking
# python3 demoGRM.py --video_path /media/phuong/DATASET/data_uav/video/bird3.mp4
