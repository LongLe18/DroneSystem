import sys
import threading
sys.path.append("..")

from PyQt5 import uic
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, Qt
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QListWidgetItem, QHBoxLayout, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
import PySpin

import cv2
import time
from PIL import Image
from rt_detector import Detector
from rt_tracker import Tracker
from sahi.utils.cv import (
    cv2,
    read_image_as_pil,
    draw_history_path,
    check_tracker,
    draw_dets,
    draw_sight,
)
from sahi.predict import get_sliced_prediction, get_prediction

system = PySpin.System.GetInstance()
# Get current library version
version = system.GetLibraryVersion()
cam_list = system.GetCameras()
print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

                
class MessageBox():
    def __init__(self, message):
        self.msg_box = QMessageBox()
        self.msg_box.setWindowTitle("Warning")
        self.msg_box.setText(message)
    
    def warning(self):
        self.msg_box.setIcon(QMessageBox.Warning)
        self.msg_box.exec_()

    def information(self):
        self.msg_box.setIcon(QMessageBox.Information)
        self.msg_box.exec_()

class ObjectListItem(QWidget):
    def __init__(self, image, name, index, accuracy):
        super().__init__()
        self.index = index

        layout = QHBoxLayout()

        image_label = QLabel(self)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        image_label.setPixmap(QPixmap.fromImage(qImg))
        layout.addWidget(image_label)

        info_layout = QVBoxLayout()  # New layout for info (name, accuracy, time)

        self.name_label = QLabel(name, self)
        self.name_label.setFont(QFont('Arial', 12))
        info_layout.addWidget(self.name_label)

        self.accuracy_label = QLabel(f'Độ chính xác: {accuracy*100:.2f}%', self)
        info_layout.addWidget(self.accuracy_label)

        self.time_label = QLabel(self)
        self.update_time()  # Initialize the time label with current time
        info_layout.addWidget(self.time_label)

        layout.addLayout(info_layout)
        self.setLayout(layout)


    def update_time(self):
        current_time = QDateTime.currentDateTime().toString('HH:mm:ss')
        self.time_label.setText(f'Thời gian: {current_time}')

    def get_name(self):
        return self.name_label.text()
    
    def get_index(self):
        return self.index
    
class CameraDialog(QtWidgets.QDialog):

    def __init__(self, cap, detector, tracker, selectSlicing, updateList):
        super().__init__()

        self.detector = detector
        self.tracker = tracker
        self.selectSlicing = selectSlicing
        self.updateList = updateList
        self.frame_count = -1
        self.counter = 0
        self.frame_time = 0.03
        self.image_dim = (1000, 800)
        self.detections = []
        self.s_tracker = None
        self.single_track_mode = False

        self.setWindowTitle("Camera Feed")

        # Create a QLabel for displaying the camera feed
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 1000, 800))  # Set the size

        # Start the camera feed in the label
        self.cap = cap
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(30)

        self.lastFrameTime = time.time()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            #close this programm
            self.close()
        if e.key() == Qt.Key_R:
            # change target
            self.tracker.tracker.change_selected_track()
            self.single_track_mode = False
            self.s_tracker = None
            self.tracker.root_s_tracker.h_path = []
            
        if e.key() == Qt.Key_F:
            # change target
            self.single_track_mode = True

    def processVideo(self, frame):
        image_as_pil = read_image_as_pil(frame)
        image = np.ascontiguousarray(image_as_pil)
        image_copy = np.copy(image)
        h, w, _ = image.shape
        
        crop_drone = None
        drone_ctx = None
        drone_cty = None

        self.frame_count += 1

        self.single_track_mode = True
        if (self.single_track_mode):
            if(self.s_tracker == None): # init new single tracker
                if len(self.detections) > 0:
                    obj_for_tracking = self.detections[0]
                    bbox = obj_for_tracking.bbox.to_xyxy() # (xmin, ymin) , (xmax, ymax)
                    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    self.s_tracker = self.tracker.create_tracker()
                    self.s_tracker.initialize(image_copy, {'init_bbox': [p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]]}) # (xmin, ymin, width, height)
                else:
                    self.single_track_mode = False
                    self.s_tracker = None      
            else: # update existing single tracker
                
                out = self.s_tracker.track(image_copy)
                state = [int(s) for s in out['target_bbox']]
                # self.tracker.root_s_tracker.h_path.append((state[0], state[1], state[2], state[3]))

                temp = image_copy[int(state[1]) : int(state[1] + state[3]), int(state[0]) : int(state[0] + state[2]), :]
                if temp.shape[0] > 0 and temp.shape[1] > 0:
                    crop_drone = temp
                    crop_drone = cv2.resize(crop_drone, (100, 80))
                    ctx = state[0] + state[2]/2
                    cty = state[1] + state[3]/2
                    # dx = float(ctx - w / 2) / w
                    # dy = float(cty - h / 2) / h
                    # gimbal_motion((dx, dy))
                    cv2.line(image_copy, (int(w / 2), int(h / 2)), (int(ctx), int(cty)), (0, 255, 255) ,2) # blue
                    cv2.rectangle(image_copy, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 0, 255), 2)

                    # draw_history_path(image_copy, self.tracker.root_s_tracker.h_path, num_point = 50)
                    drone_ctx = ctx
                    drone_cty = cty
                else:
                    self.single_track_mode = False
                    self.s_tracker = None
                    print('---------------------lost object------------------')
        if self.frame_count % 50 == 0 or self.single_track_mode == False or self.s_tracker == None:
            # perform prediction
            if self.selectSlicing == 'None':
                prediction_result = get_prediction(
                    image=image_as_pil,
                    detection_model=self.detector.detection_model,
                    shift_amount=[0, 0],
                    full_shape=None,
                    postprocess=None,
                    verbose=0,
                )
            else:
                prediction_result = get_sliced_prediction(
                    image=image_as_pil,
                    detection_model=self.detector.detection_model,
                    slice_height=int(self.selectSlicing),
                    slice_width=int(self.selectSlicing),
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=not False,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=False,
                    verbose=1 if 1 else 0,
                )

            dets = prediction_result.object_prediction_list
            self.detections = dets
            
            if drone_cty != None and len(dets) > 0:
                if not check_tracker((drone_ctx, drone_cty), dets, thresh=5): # check_tracker return false
                    self.counter += 1
                    if self.counter > 5: # neu dang track object co toa do khac xa toa do detect duoc qua 5 frame
                        self.s_tracker = None
                        self.counter = 0
                        print('==========================delete tracker===================')
                else:
                    # pass
                    self.counter = 0

            self.tracker.tracker.update(dets)
            self.updateList(dets, image)
            image_copy = draw_dets(image_copy, dets)
        
        if crop_drone is not None:
            image_copy[20:100, 880:980] = crop_drone

        # fps
        end = time.time()
        self.frame_time = self.frame_time + (end - self.lastFrameTime - self.frame_time) / 30.0
        self.lastFrameTime = end
        fps  = "FPS: {:.2f}".format(1.0 / self.frame_time)
        image_copy = draw_sight(image_copy, int(self.image_dim[0 ]/ 2), int(self.image_dim[1] / 2))
        cv2.putText(image_copy, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image_copy
    
    def update_camera_feed(self):

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.image_dim, interpolation = cv2.INTER_AREA)

            # process frame with model
            image_path = Image.fromarray(frame)
            frame = self.processVideo(image_path)

            height, width, channel = frame.shape
            bytes_per_line = channel * width
            qImg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qImg))
        else:
            self.close()

class CameraDialogIR(QtWidgets.QDialog):
    def __init__(self, cam, detector, tracker, selectSlicing, updateList):
        super().__init__()

        self.detector = detector
        self.tracker = tracker
        self.selectSlicing = selectSlicing
        self.updateList = updateList
        self.frame_count = -1
        self.counter = 0
        self.frame_time = 0.03
        self.image_dim = (1000, 800)
        self.detections = []
        self.s_tracker = None
        self.single_track_mode = False
        self.setWindowTitle("Camera Feed IR")

        # Create a QLabel for displaying the camera feed
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 1000, 800))  # Set the size

        # Start the camera feed in the label
        self.cam = cam
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

        self.lastFrameTime = time.time()

    def processVideo(self, frame):
        image_as_pil = read_image_as_pil(frame)
        image = np.ascontiguousarray(image_as_pil)
        image_copy = np.copy(image)
        h, w, _ = image.shape
        
        crop_drone = None
        drone_ctx = None
        drone_cty = None

        self.frame_count += 1

        self.single_track_mode = True
        if (self.single_track_mode):
            if(self.s_tracker == None): # init new single tracker
                if len(self.detections) > 0:
                    obj_for_tracking = self.detections[0]
                    bbox = obj_for_tracking.bbox.to_xyxy() # (xmin, ymin) , (xmax, ymax)
                    p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    self.s_tracker = self.tracker.create_tracker()
                    self.s_tracker.initialize(image_copy, {'init_bbox': [p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]]}) # (xmin, ymin, width, height)
                else:
                    self.single_track_mode = False
                    self.s_tracker = None      
            else: # update existing single tracker
                
                out = self.s_tracker.track(image_copy)
                state = [int(s) for s in out['target_bbox']]
                # self.tracker.root_s_tracker.h_path.append((state[0], state[1], state[2], state[3]))

                temp = image_copy[int(state[1]) : int(state[1] + state[3]), int(state[0]) : int(state[0] + state[2]), :]
                if temp.shape[0] > 0 and temp.shape[1] > 0:
                    crop_drone = temp
                    crop_drone = cv2.resize(crop_drone, (100, 80))
                    ctx = state[0] + state[2]/2
                    cty = state[1] + state[3]/2
                    # dx = float(ctx - w / 2) / w
                    # dy = float(cty - h / 2) / h
                    # gimbal_motion((dx, dy))
                    cv2.line(image_copy, (int(w / 2), int(h / 2)), (int(ctx), int(cty)), (0, 255, 255) ,2) # blue
                    cv2.rectangle(image_copy, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 0, 255), 2)

                    # draw_history_path(image_copy, self.tracker.root_s_tracker.h_path, num_point = 50)
                    drone_ctx = ctx
                    drone_cty = cty
                else:
                    self.single_track_mode = False
                    self.s_tracker = None
                    print('---------------------lost object------------------')
        if self.frame_count % 50 == 0 or self.single_track_mode == False or self.s_tracker == None:
            # perform prediction
            if self.selectSlicing == 'None':
                prediction_result = get_prediction(
                    image=image_as_pil,
                    detection_model=self.detector.detection_model,
                    shift_amount=[0, 0],
                    full_shape=None,
                    postprocess=None,
                    verbose=0,
                )
            else:
                prediction_result = get_sliced_prediction(
                    image=image_as_pil,
                    detection_model=self.detector.detection_model,
                    slice_height=int(self.selectSlicing),
                    slice_width=int(self.selectSlicing),
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=not False,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=False,
                    verbose=1 if 1 else 0,
                )

            dets = prediction_result.object_prediction_list
            self.detections = dets
            
            if drone_cty != None and len(dets) > 0:
                if not check_tracker((drone_ctx, drone_cty), dets, thresh=5): # check_tracker return false
                    self.counter += 1
                    if self.counter > 5: # neu dang track object co toa do khac xa toa do detect duoc qua 5 frame
                        self.s_tracker = None
                        self.counter = 0
                        print('==========================delete tracker===================')
                else:
                    # pass
                    self.counter = 0

            self.tracker.tracker.update(dets)
            self.updateList(dets, image)
            image_copy = draw_dets(image_copy, dets)
        
        if crop_drone is not None:
            image_copy[20:100, 880:980] = crop_drone

        # fps
        end = time.time()
        self.frame_time = self.frame_time + (end - self.lastFrameTime - self.frame_time) / 30.0
        self.lastFrameTime = end
        fps  = "FPS: {:.2f}".format(1.0 / self.frame_time)
        image_copy = draw_sight(image_copy, int(self.image_dim[0 ]/ 2), int(self.image_dim[1] / 2))
        cv2.putText(image_copy, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image_copy
    
    def update_camera(self):
        image = self.capture_image()
        detected_image = self.processThermalCamera(image)
        q_detected_image = self.convert_image_to_qimage(detected_image)
        self.label.setPixmap(QPixmap.fromImage(q_detected_image))

    def processThermalCamera(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, self.image_dim, interpolation = cv2.INTER_AREA) # resize image (1000, 800)

        # handle frame of video with model
        image_path = Image.fromarray(image)
        image = self.processVideo(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def capture_image(self):
        img = self.cam.GetNextImage()  # Capture an image with a timeout of 1000 milliseconds
        image_data = img.GetNDArray()  # Get image data as numpy array
        img.Release()  # Release the image to free resources
        return image_data
    
    def convert_image_to_qimage(self, image):
        height, width = image.shape
        q_image = QImage(image.data, width, height, QImage.Format_Grayscale8)
        return q_image
    
class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("ui/mainwindow.ui", self)

        # Init Detector
        self.detector = Detector()

        # Init Tracker
        self.tracker = Tracker()
        self.tracker.init_model()

        self.timerCamera1 = QTimer()
        self.timerCamera1.timeout.connect(self.getVideo)
        self.timerCamera2 = QTimer()
        self.timerCamera2.timeout.connect(self.getVideo)
        self.timerCamera3 = QTimer()
        self.timerCamera3.timeout.connect(self.update_camera)
        self.timerCamera4 = QTimer()
        self.timerCamera4.timeout.connect(self.getVideo)
        self.timerCamera5 = QTimer()
        self.timerCamera5.timeout.connect(self.getVideo)
        self.timerCamera6 = QTimer()
        self.timerCamera6.timeout.connect(self.getVideo)
        self.timerCamera7 = QTimer()
        self.timerCamera7.timeout.connect(self.getVideo)
        self.timerCamera8 = QTimer()
        self.timerCamera8.timeout.connect(self.getVideo)
        self.timerCamera9 = QTimer()
        self.timerCamera8.timeout.connect(self.getVideo)


        # set control_bt callback clicked  function
        self.bt_turnoffcamera.clicked.connect(self.reset_cameras)
        self.bt_video_main.clicked.connect(self.run_video_main)
        self.bt_video_test.clicked.connect(self.run_video_test)
        self.bt_video_thermal.clicked.connect(self.run_video_thermal)
        self.blank_image = np.zeros((1000, 800, 3), np.uint8)

        self.camera_labels = {
            0: self.label_1,
            1: self.label_2,
            2: self.label_3,
            3: self.label_4,
            4: self.label_5,
            5: self.label_6,
            6: self.label_7,
            7: self.label_8,
            8: self.label_9,
        }
        self.camera_timers = {
            0: self.timerCamera1,
            1: self.timerCamera2,
            2: self.timerCamera3,
            3: self.timerCamera4,
            4: self.timerCamera5,
            5: self.timerCamera6,
            6: self.timerCamera7,
            7: self.timerCamera8,
            8: self.timerCamera9,
        }
        self.active_cameras = set()  # To keep track of active cameras
        self.cap = {}

        self.label_menus = {}  # Dictionary to store context menus for each label
        for label_index, label_widget in self.camera_labels.items():
            label_context_menu = QtWidgets.QMenu(self)
            action_tool_1 = label_context_menu.addAction("Phát hiện")
            action_tool_2 = label_context_menu.addAction("Tạm dừng")

            action_tool_1.setData(label_index)
            action_tool_2.setData(label_index)


            label_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            label_widget.customContextMenuRequested.connect(self.show_label_context_menu)

            # Store the label and context menu in a dictionary
            self.label_menus[label_widget] = label_context_menu

        self.lastFrameTime = time.time()    
    
    def updateList(self, list_objects, image):
        self.object_list_widget.clear()

        for idx, obj in enumerate(list_objects):
            bbox = obj.bbox.to_xyxy() # (xmin, ymin) , (xmax, ymax)
            temp = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :]
            score = obj.score.value
            crop_drone = cv2.resize(temp, (100, 80))

            custom_widget = ObjectListItem(crop_drone, obj.category.name, idx, score)
            item = QListWidgetItem(self.object_list_widget)
            item.setSizeHint(custom_widget.sizeHint())
            self.object_list_widget.addItem(item)
            self.object_list_widget.setItemWidget(item, custom_widget)

    def getVideo(self): 
        for camera_idx, cap in self.cap.copy().items():
            if camera_idx == 2: continue
            ret, image = cap.read()
            if(ret == False):
                image = self.blank_image 
                self.camera_timers[camera_idx].stop()
                self.active_cameras.remove(camera_idx)
                self.cap[camera_idx].release()
                self.cap.pop(camera_idx)
            
            # convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (295, 265), interpolation = cv2.INTER_AREA)

            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            label_widget = self.camera_labels[camera_idx]
            label_widget.setPixmap(QPixmap.fromImage(qImg))

    def reset_cameras(self):
        for camera_idx in self.active_cameras:
            self.cap[camera_idx].release()
            self.camera_labels[camera_idx].clear()

        self.active_cameras.clear()

    def run_video_test(self):
        idx = 0
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        video_file, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
    
        if video_file:
            print(f"Selected video file: {video_file}")
            cap = cv2.VideoCapture(video_file)
            for camera_idx, label_widget in self.camera_labels.items():
                if camera_idx in self.active_cameras or camera_idx == 2:
                    continue
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    self.active_cameras.add(camera_idx)
                    self.cap[camera_idx] = cap
                    idx = camera_idx
                    break
            self.camera_timers[idx].start(30)
        
    def run_video_main(self):
        idx = 0
        for camera_idx, label_widget in self.camera_labels.items():
            if camera_idx in self.active_cameras:
                idx += 1
                continue
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.active_cameras.add(camera_idx)
                self.cap[camera_idx] = cap
                idx = camera_idx
                break
        self.camera_timers[idx].start(30)


    """
    Handle Camera thermal (FLIR Ax5)
    """
    def acquire_and_display_images(self, nodemap, nodemap_tldevice):
        sNodemap = self.cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            msg_box = MessageBox('Unable to set stream buffer handling mode.. Aborting...')
            msg_box.warning()

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(node_newestonly):
            msg_box = MessageBox('Unable to set stream buffer handling mode.. Aborting...')
            msg_box.warning()
        
        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                msg_box = MessageBox('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                msg_box.warning()

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                msg_box = MessageBox('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                msg_box.warning()

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            self.cam.BeginAcquisition() # Start the camera stream
            print('Acquiring images...')

            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

    def initialize_camera(self):
        # Retrieve list of cameras from the system
        num_cameras = cam_list.GetSize()
        print('Number of cameras detected: %d' % num_cameras)
        
        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            cam_list.Clear()
            # Release system instance
            system.ReleaseInstance()
            msg_box = MessageBox("Không có camera!")
            msg_box.warning()

        self.cam = cam_list.GetByIndex(0)  # Assuming the first camera
        self.cam.Init()
        nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        nodemap = self.cam.GetNodeMap()
        self.acquire_and_display_images(nodemap, nodemap_tldevice)

    def update_camera(self):
        image = self.capture_image()
        detected_image = self.processThermalCamera(image)
        q_detected_image = self.convert_image_to_qimage(detected_image)
        self.label_3.setPixmap(QPixmap.fromImage(q_detected_image))

    def processThermalCamera(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (295, 265), interpolation = cv2.INTER_AREA) # resize image (1000, 800)

        # handle frame of video with model
        # image_path = Image.fromarray(image)
        # image = self.processVideo(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def capture_image(self):
        img = self.cam.GetNextImage()  # Capture an image with a timeout of 1000 milliseconds
        image_data = img.GetNDArray()  # Get image data as numpy array
        img.Release()  # Release the image to free resources
        return image_data
    
    def convert_image_to_qimage(self, image):
        height, width = image.shape
        q_image = QImage(image.data, width, height, QImage.Format_Grayscale8)
        return q_image
    
    def run_video_thermal(self):
        # if timer is stopped
        if self.bt_video_thermal.text() == "Camera nhiệt":

            self.initialize_camera()  # Initialize the camera and start the stream
 
            self.active_cameras.add(2)
            self.cap[2] = 'ir'
            self.camera_timers[2].start(30)
            # start timer
            self.bt_video_thermal.setText("Dừng stream")
            self.frame_count = -1

        else:
            # stop timer
            self.camera_timers[2].stop()
            # update control_bt text
            self.bt_video_thermal.setText("Camera nhiệt")

            self.cam.EndAcquisition()  # End the camera stream
            self.cam.DeInit()  # Deinitialize the camera
            del self.cam
            
    """
    END Functions for FLIR camera
    """

    def open_camera_dialog(self, camera_index, ir=False):
        print("Opening camera dialog")
        if ir:
            camera_dialog = CameraDialogIR(camera_index, detector=self.detector, tracker=self.tracker, 
                                         selectSlicing=self.selectSlicing.currentText(), updateList=self.updateList)
        else: 
            camera_dialog = CameraDialog(camera_index, detector=self.detector, tracker=self.tracker,
                                         selectSlicing=self.selectSlicing.currentText(), updateList=self.updateList)
        camera_dialog.show()
        camera_dialog.exec_()
    
    def show_label_context_menu(self, pos):
        label = self.sender()  # Get the label that triggered the event
        if label in self.label_menus:
            label_context_menu = self.label_menus[label]

            # Show the context menu at the mouse cursor position
            global_pos = label.mapToGlobal(pos)
            selected_action = label_context_menu.exec_(global_pos)

            if selected_action:
                selected_label_index = selected_action.data()
                if selected_action == label_context_menu.actions()[0]:  # Tool 1 action
                    print(f"Tool 1 chosen for label {selected_label_index}")
                    if (int(selected_label_index) == 2):
                        self.open_camera_dialog(self.cam, True)
                    else:
                        # stop timer main window
                        self.camera_timers[int(selected_label_index)].stop()
                        # open dialog for object detection
                        self.open_camera_dialog(self.cap[int(selected_label_index)])

                elif selected_action == label_context_menu.actions()[1]:  # Tool 2 action
                    if (int(selected_label_index) == 2):
                        # stop timer
                        self.camera_timers[2].stop()
                        # update control_bt text
                        self.bt_video_thermal.setText("Camera nhiệt")

                        self.cam.EndAcquisition()  # End the camera stream
                        self.cam.DeInit()  # Deinitialize the camera
                        del self.cam
                    else:
                        self.camera_timers[int(selected_label_index)].stop()
                        self.active_cameras.remove(int(selected_label_index))
                        self.cap[int(selected_label_index)].release()
                        self.cap.pop(int(selected_label_index))
                    print(f"Tool 2 chosen for label {selected_label_index}")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    # mainWindow = uic.loadUi("mainwindow.ui")
    mainWindow.show()
    
    sys.exit(app.exec_())
