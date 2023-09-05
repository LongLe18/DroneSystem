# ! python demo.py --type-model onnx \
#     --yolo-model weights/detection/yolov8_drone.onnx  \
#     --source /media/phuong/DATASET/data_uav/video/V_DRONE_030.mp4 \
#     --conf 0.25  \
#     --show \
#     --apply-tracking \

# demo sequence tracking
# python3 demoSeq.py seqtrack seqtrack_b256 --video_path /media/phuong/DATASET/data_uav/video/video6.mp4

# demo GRM tracking
# python3 demoGRM.py --video_path /media/phuong/DATASET/data_uav/video/bird3.mp4
