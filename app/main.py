import PySpin

# import system module
import sys
sys.path.append("..")
from PyQt5.QtCore import Qt
#create release: pyinstaller --onefile
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidgetItem, QWidget, QMessageBox
from PyQt5 import uic
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QTimer, QDateTime

# import Opencv module
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
from sahi.predict import get_sliced_prediction

global continue_recording
continue_recording = True

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

        self.accuracy_label = QLabel(f'Accuracy: {accuracy*100:.2f}%', self)
        info_layout.addWidget(self.accuracy_label)

        self.time_label = QLabel(self)
        self.update_time()  # Initialize the time label with current time
        info_layout.addWidget(self.time_label)

        layout.addLayout(info_layout)
        self.setLayout(layout)


    def update_time(self):
        current_time = QDateTime.currentDateTime().toString('HH:mm:ss')
        self.time_label.setText(f'Time: {current_time}')

    def get_name(self):
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
        
        self.timerCameraIR = QTimer()
        self.timerCameraIR.timeout.connect(self.update_camera)
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
        self.bt_video_thermal.clicked.connect(self.run_video_thermal)
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

        self.videoWritter.write(image)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
    
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
            score = obj.score.value
            crop_drone = cv2.resize(temp, (100, 80))

            custom_widget = ObjectListItem(crop_drone, obj.category.name, idx, score)
            item = QListWidgetItem(self.object_list_widget)
            item.setSizeHint(custom_widget.sizeHint())
            self.object_list_widget.addItem(item)
            self.object_list_widget.setItemWidget(item, custom_widget)

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
            msg_box = MessageBox("Not enough cameras!")
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
        self.image_label.setPixmap(QPixmap.fromImage(q_detected_image))

    def processThermalCamera(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, self.image_dim, interpolation = cv2.INTER_AREA) # resize image (1000, 800)

        # handle frame of video with model
        image_path = Image.fromarray(image)
        image = self.processVideo(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # write video
        self.videoWritter.write(image)
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
        if not self.timerCameraIR.isActive():

            self.initialize_camera()  # Initialize the camera and start the stream

            # start timer
            self.timerCameraIR.start(30)
            self.bt_video_thermal.setText("Dừng streaming")
            self.frame_count = -1

        else:
            # stop timer
            self.timerCameraIR.stop()
            # update control_bt text
            self.bt_video_thermal.setText("Camera nhiệt")

            self.cam.EndAcquisition()  # End the camera stream
            self.cam.DeInit()  # Deinitialize the camera
            del self.cam
            
    """
    END Functions for FLIR camera
    """

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

    


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    # mainWindow = uic.loadUi("mainwindow.ui")
    mainWindow.show()
    
    sys.exit(app.exec_())
