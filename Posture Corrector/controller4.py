'''
Integrating mediapipe with PyQt
Set QLabel as output
Make function for QAction: Mute & PopUp
Make function for QAction: Select Notification & About
Debugging selectNotification function
Make Graph for Accuracy and Status 
Integrating all trigger function to newest UI
Integrating graph to function
Make function for tutorial
'''
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import cv2
import numpy as np
import time
import mediapipe as mp
# from playsound import playsound
import pygame
import pyqtgraph as pg
import pandas as pd

from FIX_display5 import Ui_MainWindow

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        pygame.mixer.init() #For Notification

        #QActionTrigger
        self.ui.streaming.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.streaming.setScaledContents(True)
        self.ui.actionCamera_ON.triggered.connect(self.startCamera)
        self.ui.actionMute.triggered.connect(self.Mute)
        self.ui.actionPop_up_Mode.triggered.connect(self.PopUp)
        self.ui.actionSelect_Notification.triggered.connect(self.selectNotification)
        self.ui.actionAbout.triggered.connect(self.About)
        self.ui.actionTutorial.triggered.connect(self.Tutorial)
        self.ui.actionExport_to_Excel.triggered.connect(self.Export_to_Excel)

        self.camera_active = False
        self.start_time = None #Representasi waktu

        self.is_muted = False #Flag mode mute
        self.popup_mode = False #Flag mode pop up

        # Posture Detection Initialization
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []
        self.calibration_nose_distance = []
        self.shoulder_threshold = 0
        self.neck_threshold = 0
        self.nose_threshold = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5
        self.sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"

        #Graph Initialization
        #Graph Status
        self.graph=pg.PlotWidget()
        self.graph.setLabel('left', 'Posture Status')  # Label sumbu Y
        self.graph.setLabel('bottom', 'Time (frame)')  # Label sumbu X

        self.graph_status=self.findChild(QWidget, "graphStatus")
        self.graph.setParent(self.graph_status)
        layout_status=QVBoxLayout(self.graph_status)
        layout_status.addWidget(self.graph)
        self.graph_status.setLayout(layout_status)
        self.status_x=[]
        self.status_y=[]
        self.status_curve=self.graph.plot(pen=pg.mkPen(color=(255,0,0),width=2))
        self.current_posture_status = 0  # 0 = Poor, 1 = Good


        #Graph Accuracy
        self.graph2=pg.PlotWidget()
        self.graph2.setLabel('left', 'Accuracy (%)')   # Label sumbu Y
        self.graph2.setLabel('bottom', 'Time (frame)') # Label sumbu X

        self.graph_acc=self.findChild(QWidget,"graphAkurasi")
        self.graph2.setParent(self.graph_acc)
        layout_acc=QVBoxLayout(self.graph_acc)
        layout_acc.addWidget(self.graph2)
        self.graph_acc.setLayout(layout_acc)
        self.acc_x=[]
        self.acc_y=[]
        self.acc_curve=self.graph2.plot(pen=pg.mkPen(color=(0,255,0),width=2))
        self.current_accuracy = 100.0  # default 100%

        #Excel log
        self.posture_log = []  # Format: (timestamp, status_string, accuracy)


        self.Tutorial() #Panggil fungsi tutorial agar muncul pertama kali ketika program dijalankan



    def update_plot_status(self): #Clear
        # if len(self.status_x) > 100:
        #     self.status_x=self.status_x[1:]
        #     self.status_y=self.status_y[1:]
        
        # new_status_x=self.status_x[-1] + 1 if self.status_x else 0 
        # new_status_y = self.current_posture_status

        # self.status_x.append(new_status_x)
        # self.status_y.append(new_status_y)
        # self.status_curve.setData(self.status_x, self.status_y)

        if len(self.status_x) > 100:
            self.status_x = self.status_x[1:]
            self.status_y = self.status_y[1:]

        new_status_x = self.status_x[-1] + 1 if self.status_x else 0
        new_status_y = self.current_posture_status

        self.status_x.append(new_status_x)
        self.status_y.append(new_status_y)
        self.status_curve.setData(self.status_x, self.status_y)

        # Log waktu dan status untuk Excel
        current_time = time.strftime('%H:%M:%S')
        status_str = "Good" if self.current_posture_status == 1 else "Poor"
        self.posture_log.append((current_time, status_str, round(self.current_accuracy, 2)))
     
    def update_plot_acc(self):
        if len(self.acc_x) > 100:
            self.acc_x=self.acc_x[1:]
            self.acc_y=self.acc_y[1:]
    
        new_acc_x=self.acc_x[-1] + 1 if self.acc_x else 0 
        new_acc_y=self.current_accuracy

        self.acc_x.append(new_acc_x)
        self.acc_y.append(new_acc_y)
        self.acc_curve.setData(self.acc_x, self.acc_y)

    

    def startCamera(self):
        if not self.camera_active:
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.timer = QTimer()
            self.timer.timeout.connect(self.updateFrame)
            self.timer.start(30)

            self.start_time = time.time()  # Mulai hitung waktu
            self.camera_active = True
            self.ui.actionCamera_ON.setText("Cam: OFF")

            #Graph Status Timer
            self.timer_status=pg.QtCore.QTimer()
            self.timer_status.timeout.connect(self.update_plot_status)
            self.timer_status.start(100) #100

            #Graph Acc Timer
            self.timer_acc=pg.QtCore.QTimer()
            self.timer_acc.timeout.connect(self.update_plot_acc)
            self.timer_acc.start(100)

        else:
            self.timer.stop()
            self.capture.release()
            self.ui.streaming.clear()
            self.ui.status.clear()
            self.ui.time.clear()
            self.ui.shoulder.clear()
            self.ui.neck.clear()
            self.ui.nose.clear()
            self.ui.actionCamera_ON.setText("Cam: ON")
            self.camera_active = False
            
            #Graph Status
            self.timer_status.stop()
            self.status_x.clear()
            self.status_y.clear()
            self.status_curve.clear()
            self.status_curve.setData([], [])  # Reset plot

            #Graph Acc
            self.timer_acc.stop()
            self.acc_x.clear()
            self.acc_y.clear()
            self.acc_curve.clear()
            self.acc_curve.setData([], [])  # Reset plot


    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
        ba, bc = a - b, c - b
        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return np.nan
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def play_alert_sound(self):      
        if not self.is_muted:
            try:
                pygame.mixer.music.load(self.sound_file)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"Error playing sound: {e}")

    def updateFrame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        status = "Calibrating..."

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Get important landmarks
            left_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                               landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y], [w, h]).astype(int))
            right_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y], [w, h]).astype(int))
            left_ear = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x,
                                          landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y], [w, h]).astype(int))
            nose = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.NOSE].x,
                                      landmarks[self.mp_pose.PoseLandmark.NOSE].y], [w, h]).astype(int))
            midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
                        (left_shoulder[1] + right_shoulder[1]) // 2)

            shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
            neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
            nose_to_midpoint_distance = np.linalg.norm(np.array(nose) - np.array(midpoint))

            if not self.is_calibrated and self.calibration_frames < 30:
                self.calibration_shoulder_angles.append(shoulder_angle)
                self.calibration_neck_angles.append(neck_angle)
                self.calibration_nose_distance.append(nose_to_midpoint_distance)
                self.calibration_frames += 1
            elif not self.is_calibrated:
                self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 5
                self.neck_threshold = np.mean(self.calibration_neck_angles) - 5
                self.nose_threshold = np.mean(self.calibration_nose_distance) - 10
                self.is_calibrated = True

            if self.is_calibrated:
                current_time = time.time()
                
                # Hitung akurasi berdasarkan seberapa dekat nilai saat ini ke ambang batas
                shoulder_score = min(shoulder_angle / self.shoulder_threshold, 1.0)
                neck_score = min(neck_angle / self.neck_threshold, 1.0)
                nose_score = min(nose_to_midpoint_distance / self.nose_threshold, 1.0)

                # Ambil rata-rata dan ubah jadi persentase
                accuracy = (shoulder_score + neck_score + nose_score) / 3 * 100
                self.current_accuracy = accuracy


                if (shoulder_angle < self.shoulder_threshold or
                    neck_angle < self.neck_threshold or
                    nose_to_midpoint_distance < self.nose_threshold):
                    status = "Poor Posture"
                    self.current_posture_status = 0 #Flag for graph
                    color = (0, 0, 255)
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        self.play_alert_sound()
                        self.last_alert_time = current_time
                        if self.popup_mode:
                            QMessageBox.warning(self, "Posture Alert", "Warning: Poor posture detected!", QMessageBox.StandardButton.Ok)

                else:
                    status = "Good Posture"
                    self.current_posture_status = 1 #Flag for graph
                    color = (0, 255, 0)

                for pt in [left_shoulder, right_shoulder, left_ear, nose, midpoint]:
                    cv2.circle(frame, pt, 8, color, -1)

                # Update QLabel status
                self.ui.status.setText(f"{status}")

                # Update QLabel Time
                # elapsed_time = int(time.time() - self.start_time)
                # self.ui.time.setText(f"Elapsed Time: {elapsed_time} s")

                elapsed_time = int(time.time() - self.start_time)
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                self.ui.time.setText(f"Elapsed Time: {formatted_time}")

                
                # Update QLabel Degree
                # self.ui.degree.setText(
                #     f"Shoulder: {shoulder_angle:.1f}° (Threshold: {self.shoulder_threshold:.1f}°)\n"
                #     f"Neck: {neck_angle:.1f}° (Threshold: {self.neck_threshold:.1f}°)\n"
                #     f"Nose Dist: {nose_to_midpoint_distance:.1f} (Threshold: {self.nose_threshold:.1f})"
                # )

                #Update QLabel Shoulder Angle
                self.ui.shoulder.setText(f"Shoulder: {shoulder_angle:.1f}° (Threshold: {self.shoulder_threshold:.1f}°)")

                #Update QLabel Neck Angle
                self.ui.neck.setText(f"Neck: {neck_angle:.1f}° (Threshold: {self.neck_threshold:.1f}°)")

                #Update QLabel Nose Distance
                self.ui.nose.setText(f"Nose Dist: {nose_to_midpoint_distance:.1f} (Threshold: {self.nose_threshold:.1f})")


            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Convert to QImage and show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        scaled_img = qt_img.scaled(self.ui.streaming.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                   Qt.TransformationMode.SmoothTransformation)
        label_size = self.ui.streaming.size()
        rect = QRect(
            (scaled_img.width() - label_size.width()) // 2,
            (scaled_img.height() - label_size.height()) // 2,
            label_size.width(),
            label_size.height()
        )
        cropped_img = scaled_img.copy(rect)
        self.ui.streaming.setPixmap(QPixmap.fromImage(cropped_img))

    def Mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.ui.actionMute.setText("Unmute")
            self.ui.statusbar.showMessage("Alert sound muted", 3000)
        else:
            self.ui.actionMute.setText("Mute")
            self.ui.statusbar.showMessage("Alert sound unmuted", 3000)

    def PopUp(self):
        self.popup_mode = not self.popup_mode
        if self.popup_mode:
            self.ui.actionPop_up_Mode.setText("Pop-up: ON")
            self.ui.statusbar.showMessage("Pop-up mode enabled", 3000)
        else:
            self.ui.actionPop_up_Mode.setText("Pop-up: OFF")
            self.ui.statusbar.showMessage("Pop-up mode disabled", 3000)

    def selectNotification(self):
        file_path, _ = QFileDialog.getOpenFileName(
        self,
        "Select Notification Sound",
        "",
        "Audio Files (*.mp3 *.wav *.ogg);;All Files (*)"
        )
        if file_path:
            self.sound_file = file_path
            self.ui.statusbar.showMessage(f"Notification sound selected: {file_path}", 3000)
    
    # def About(self):
    #     dialog = QDialog(self)
    #     dialog.setWindowTitle("About")
    #     dialog.setFixedSize(300, 200)

    #     layout = QVBoxLayout()

    #     layout.addWidget(QLabel("<h3>Posture Monitor</h3>"))
    #     layout.addWidget(QLabel("Version: 1.0.0"))
    #     layout.addWidget(QLabel("Author: Nama Kamu"))
    #     layout.addWidget(QLabel("Made with ❤️ using PyQt and MediaPipe"))

    #     close_button = QPushButton("Close")
    #     close_button.clicked.connect(dialog.close)
    #     layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)

    #     dialog.setLayout(layout)
    #     dialog.exec()

    def About(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About")
        dialog.setFixedSize(400, 250)

        layout = QVBoxLayout()

        # Title with style
        title = QLabel("<h2 style='color: #2E86C1;'>Posture Monitoring App</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description with HTML
        info = QLabel("""
            <p align='center'>
                Version: <b>1.0.0</b><br>
                Developed by: <b>DAR</b><br><br>
                A real-time posture monitoring application using<br>
                <span style='color: #27AE60;'>MediaPipe</span>, <span style='color: #F39C12;'>OpenCV</span>, and <span style='color: #8E44AD;'>PyQt6</span>.
            </p>
        """)
        info.setWordWrap(True)
        layout.addWidget(info)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.setStyleSheet("padding: 6px; background-color: #3498DB; color: white; border-radius: 6px;")
        close_btn.clicked.connect(dialog.accept)

        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec()

    def Tutorial(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Tutorial")
        dialog.setFixedSize(450, 300)

        layout = QVBoxLayout()

        title = QLabel("<h2 style='color:#2980B9;'>Welcome to Posture Monitor!</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        content = QLabel("""
            <p>
            This tutorial will help you get started:
            <ul>
                <li><b>Camera ON:</b> Start real-time posture monitoring.</li>
                <li><b>Mute:</b> Enable/disable alert sound.</li>
                <li><b>Pop-up Mode:</b> Show alert message when posture is bad.</li>
                <li><b>Select Notification:</b> Choose your own alert sound (.mp3/.wav).</li>
                <li><b>About:</b> View app information.</li>
                <li><b>Graphs:</b> See your posture status and accuracy in real-time.</li>
            </ul>
            </p>
            <p style='color: #16A085;'><i>Make sure to sit straight during the first 3 seconds for calibration.</i></p>
        """)
        content.setWordWrap(True)
        layout.addWidget(content)

        close_btn = QPushButton("Got it!")
        close_btn.setFixedWidth(100)
        close_btn.setStyleSheet("padding: 6px; background-color: #27AE60; color: white; border-radius: 6px;")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec()

    def Export_to_Excel(self):
        if not self.posture_log:
            QMessageBox.information(self, "No Data", "No posture data available to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Posture Report",
            "posture_report.xlsx",
            "Excel Files (*.xlsx)"
        )

        if file_path:
            try:
                df = pd.DataFrame(self.posture_log, columns=["Time", "Posture Status", "Accuracy (%)"])
                df.to_excel(file_path, index=False)
                self.ui.statusbar.showMessage(f"Data successfully exported to {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{e}")


        
    def closeEvent(self, event):
        if hasattr(self, 'capture') and self.capture is not None:
            self.timer.stop()
            self.capture.release()
            self.capture = None
        event.accept()

