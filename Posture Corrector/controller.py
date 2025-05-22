'''[BASE CODE]'''
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import cv2
from FIX_display import Ui_MainWindow
class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)

        # self.setWindowTitle("Live Camera")
        # self.setGeometry(100,100,1280,720)

        #Qlabel Configuration
        self.ui.streaming.setAlignment(Qt.AlignmentFlag.AlignCenter)

        #Trigger Menu Bar
        self.ui.actionCamera_ON.triggered.connect(self.startCamera)

        self.ui.streaming.setScaledContents(True) #Memaksa pixmap memenuhi label 
     
        self.camera_active=False


    def startCamera(self):
        if self.camera_active == False:
            # self.ui.streaming.setFixedSize(1280,720)
            self.capture=cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.timer=QTimer()
            self.timer.timeout.connect(self.updateFrame)
            self.timer.start(30)

            self.camera_active=True
            self.ui.actionCamera_ON.setText("Cam: OFF")
        
        else:
             self.timer.stop()
             self.capture.release()
             self.ui.streaming.clear()
             self.ui.actionCamera_ON.setText("Cam: ON")
             self.camera_active=False
    
    def updateFrame(self):
        ret, frame = self.capture.read()
        if ret:
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch=frame.shape
            # self.ui.streaming.setFixedSize(w, h) #Otomatis setting display kamera agar sesuai dengan frame kamera
            bytes_per_line = ch * w
            qt_img=QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Resize QImage to fit QLabel size
            # scaled_img = qt_img.scaled(self.ui.streaming.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
            # scaled_img = qt_img.scaled(self.ui.streaming.size(),Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation)

            # self.ui.streaming.setPixmap(QPixmap.fromImage(scaled_img))

            scaled_img = qt_img.scaled(self.ui.streaming.size(),Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation)

            # Crop to fit QLabel size (centered)
            label_size = self.ui.streaming.size()
            rect = QRect(
                (scaled_img.width() - label_size.width()) // 2,
                (scaled_img.height() - label_size.height()) // 2,
                label_size.width(),
                label_size.height()
            )
            cropped_img = scaled_img.copy(rect)
            self.ui.streaming.setPixmap(QPixmap.fromImage(cropped_img))

    
    
    def closeEvent(self, event):
        if hasattr(self, 'capture') and self.capture is not None:
            self.timer.stop()
            self.capture.release()
            self.capture = None
        event.accept()
