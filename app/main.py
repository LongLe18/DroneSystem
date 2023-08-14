

# import system module
import sys
sys.path.append("..")
from PyQt5.QtCore import Qt
import socket
#create release: pyinstaller --onefile
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidgetItem, QWidget
from PyQt5 import uic
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2
import math
import time
import serial
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
from sahi.predict import get_sliced_prediction

class ObjectListItem(QWidget):
    def __init__(self, image, name, index):
        super().__init__()
        self.index = index

        layout = QHBoxLayout()

        image_label = QLabel(self)
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        image_label.setPixmap(QPixmap.fromImage(qImg))
        layout.addWidget(image_label)

        self.name_label = QLabel(name, self)
        self.name_label.setFont(QFont('Arial', 14))  # Set the font size to 14
        layout.addWidget(self.name_label)

        self.setLayout(layout)

    def get_name(self):
        # Implement this method to return the name data used to create the custom widget
        # Return the name here
        return self.name_label.text()
    
    def get_index(self):
        return self.index
    
class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("ui/mainwindow.ui", self)

        self.h_speed = 0
        self.v_speed = 0

        # Init Detector
        self.detector = Detector()

        # Init Tracker
        self.tracker = Tracker()
        self.tracker.init_model()

        self.s_tracker = None
        self.single_track_mode = False

        self.detections = []
        
        self.timerCamera = QTimer()
        self.timerCamera.timeout.connect(self.getVideo)
        self.timerControl = QTimer()
        self.timerControl.start(30)
        self.image_dim = (1000, 800)
        self.blank_image = np.zeros((1000, 800, 3), np.uint8)
        self.frame_count = -1
        self.lastFrameTime = time.time()
        # set control_bt callback clicked  function
        # self.bt_video_main.clicked.connect(self.updateList)
        # self.bt_video_thermal.clicked.connect(self.run_video_thermal)
        self.bt_video_test.clicked.connect(self.run_video_test)
        self.frame_time = 0.03

        self.videoWritter = cv2.VideoWriter('out_vid_test4.mp4.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, self.image_dim)
        self.object_list_widget.itemClicked.connect(self.itemClicked)
        self.counter = 0

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            #close this programm
            self.close()
        if e.key() == Qt.Key_W:
            # move up
            self.v_speed = 255
        if e.key() == Qt.Key_S:
            #move down
            self.v_speed =- 255
        if e.key() == Qt.Key_A:
            #move left
            self.h_speed =- 255
        if e.key() == Qt.Key_D:
            # move right
            self.h_speed = 255
        if e.key() == Qt.Key_R:
            # change target
            self.tracker.tracker.change_selected_track()
            self.single_track_mode = False
            self.s_tracker = None
            self.tracker.root_s_tracker.h_path = []
            
        if e.key() == Qt.Key_F:
            # change target
            self.single_track_mode = True

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_W:
            # stop
            self.v_speed = 0
        if e.key() == Qt.Key_S:
            # stop
            self.v_speed =- 0
        if e.key() == Qt.Key_A:
            # stop
            self.h_speed =- 0
        if e.key() == Qt.Key_D:
            # stop
            self.h_speed = 0

    def reset(self):
        self.single_track_mode = False
        self.s_tracker = None      
        self.detections = []
        self.tracker.root_s_tracker.h_path = []
        
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
                self.tracker.root_s_tracker.h_path.append((state[0], state[1], state[2], state[3]))

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

                    draw_history_path(image_copy, self.tracker.root_s_tracker.h_path, num_point = 50)
                    drone_ctx = ctx
                    drone_cty = cty
                else:
                    self.single_track_mode = False
                    self.s_tracker = None
                    print('---------------------lost object------------------')
        if self.frame_count % 50 == 0 or self.single_track_mode == False or self.s_tracker == None:
            # perform prediction
            prediction_result = get_sliced_prediction(
                image=image_as_pil,
                detection_model=self.detector.detection_model,
                slice_height=int(self.selectSlicing.currentText()),
                slice_width=int(self.selectSlicing.currentText()),
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
    
    # view camera
    def getVideo(self):
        # read image in BGR format
        ret, image = self.cap.read()
        if(ret == False):
            image = self.blank_image 
            return image
        
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_dim, interpolation = cv2.INTER_AREA)
        image_path = Image.fromarray(image)
        image = self.processVideo(image_path)

        # img_cp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.videoWritter.write(image)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
    # start/stop timer
    def run_video_test(self):
        def run(video):
                # create video capture
                self.cap = cv2.VideoCapture(video)
                # start timer
                self.timerCamera.start(30)
                # update control_bt text
                self.bt_video_test.setText("Dừng video")
                self.frame_count = -1
            
        if not self.timerCamera.isActive():
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            video_file, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        
            if video_file:
                print(f"Selected video file: {video_file}")
                run(video_file)
        # if timer is stopped
        else:
            print("Dừng video")
            self.image_label.setText('Chon video / camera ')
            self.reset()
            # stop timer
            self.timerCamera.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.bt_video_test.setText("Chọn video")

    def itemClicked(self, item):
        # This slot will be called when an item in the list is clicked
        # You can use the item parameter to access the clicked item's data
        custom_widget = self.object_list_widget.itemWidget(item)
        if custom_widget:
            # Do something with the custom widget's data
            idx = custom_widget.get_index()    # Adjust this based on your custom widget's structure
            self.tracker.tracker.specific_selected_track(idx)
            self.single_track_mode = False
            self.s_tracker = None
            self.tracker.root_s_tracker.h_path = []
            print("Clicked:", str(idx))
            # Implement your desired action here

    def updateList(self, list_objects, image):
        self.object_list_widget.clear()

        for idx, obj in enumerate(list_objects):
            bbox = obj.bbox.to_xyxy() # (xmin, ymin) , (xmax, ymax)
            temp = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :]
            crop_drone = cv2.resize(temp, (100, 80))

            custom_widget = ObjectListItem(crop_drone, obj.category.name, idx)
            item = QListWidgetItem(self.object_list_widget)
            item.setSizeHint(custom_widget.sizeHint())
            self.object_list_widget.addItem(item)
            self.object_list_widget.setItemWidget(item, custom_widget)

    

    # start/stop timer
    # def run_video_main(self):
    #     # if timer is stopped
    #     if not self.timerCamera.isActive():
    #         # create video capture
    #         self.cap = cv2.VideoCapture(0)
    #         # start timer
    #         self.timerCamera.start(30)
    #         # update control_bt text
    #         self.bt_video_test.setText("Stop")
    #     # if timer is started
    #     else:
    #         # stop timer
    #         self.timerCamera.stop()
    #         # release video capture
    #         self.cap.release()
    #         # update control_bt text
    #         self.bt_video_test.setText("Start")

    # def run_video_thermal(self):
    #     # if timer is stopped
    #     if not self.timerCamera.isActive():
    #         # create video capture
    #         self.cap = cv2.VideoCapture(1)
    #         # start timer
    #         self.timerCamera.start(30)
    #         # update control_bt text
    #         self.bt_video_thermal.setText("Stop")
    #     # if timer is started
    #     else:
    #         # stop timer
    #         self.timerCamera.stop()
    #         # release video capture
    #         self.cap.release()
    #         # update control_bt text
    #         self.bt_video_thermal.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    # mainWindow = uic.loadUi("mainwindow.ui")
    mainWindow.show()
    
    sys.exit(app.exec_())
