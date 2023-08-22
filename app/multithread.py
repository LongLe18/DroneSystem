import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

class VideoThread(QThread):
    frame_data = pyqtSignal(np.ndarray)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_data.emit(frame)
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multi-Video Display Example")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.video_labels = [QLabel(), QLabel()]
        layout.addWidget(self.video_labels[0])
        layout.addWidget(self.video_labels[1])

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        video_paths = ["/media/phuong/DATASET/data_uav/video/video1.avi", "/media/phuong/DATASET/data_uav/video/city.mp4"]
        self.video_threads = [VideoThread(video_path) for video_path in video_paths]
        for idx, thread in enumerate(self.video_threads):
            thread.frame_data.connect(lambda frame, idx=idx: self.display_frame(frame, idx))
            thread.start()

    def display_frame(self, frame, idx):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_labels[idx].setPixmap(pixmap)
        self.video_labels[idx].setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
