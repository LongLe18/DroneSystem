import sys
from PyQt5 import uic
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QPixmap, QImage

import cv2
class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("ui/mainwindow.ui", self)

        self.h_speed = 0
        self.v_speed = 0
        
        self.timerCamera1 = QTimer()
        self.timerCamera1.timeout.connect(self.getVideo)
        self.timerCamera2 = QTimer()
        self.timerCamera2.timeout.connect(self.getVideo)
        self.timerCamera3 = QTimer()
        self.timerCamera3.timeout.connect(self.getVideo)
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

    def getVideo(self): 
        for camera_idx, cap in self.cap.items():
            # read image in BGR format
            ret, image = cap.read()
            if(ret == False):
                image = self.blank_image 
                return image
            
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
        if self.bt_video_test.text() == "Chọn video":
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            video_file, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)", options=options)
        
            if video_file:
                print(f"Selected video file: {video_file}")
                for camera_idx, label_widget in self.camera_labels.items():
                    if camera_idx in self.active_cameras:
                        continue
                    cap = cv2.VideoCapture(video_file)
                    if cap.isOpened():
                        self.active_cameras.add(camera_idx)
                        self.cap[camera_idx] = cap
                        idx = camera_idx
                        break
                self.camera_timers[idx].start(30)
                self.bt_video_test.setText("Tạm Dừng")
        else:
            print("Dừng video")
            self.camera_timers[idx].stop()
            self.active_cameras.remove(idx)
            self.cap[idx].release()
            self.bt_video_test.setText("Chọn video")

    def run_video_main(self):
        idx = 0
        try:
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
        except:
            print('Lỗi')

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
                elif selected_action == label_context_menu.actions()[1]:  # Tool 2 action
                    self.camera_timers[int(selected_label_index)].stop()
                    self.active_cameras.remove(int(selected_label_index))
                    self.cap[int(selected_label_index)].release()
                    print(f"Tool 2 chosen for label {selected_label_index}")

        
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    # mainWindow = uic.loadUi("mainwindow.ui")
    mainWindow.show()
    
    sys.exit(app.exec_())
