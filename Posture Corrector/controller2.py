# '''Integrating mediapipe with PyQt'''
# from PyQt6.QtCore import *
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import *

# import cv2
# import numpy as np
# import time
# import threading
# import mediapipe as mp
# from playsound import playsound

# from FIX_display import Ui_MainWindow

# class MainController(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)

#         self.ui.streaming.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.ui.streaming.setScaledContents(True)
#         self.ui.actionCamera_ON.triggered.connect(self.startCamera)

#         self.camera_active = False

#         # Posture Detection Initialization
#         self.mp_pose = mp.solutions.pose
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#         self.is_calibrated = False
#         self.calibration_frames = 0
#         self.calibration_shoulder_angles = []
#         self.calibration_neck_angles = []
#         self.calibration_nose_distance = []
#         self.shoulder_threshold = 0
#         self.neck_threshold = 0
#         self.nose_threshold = 0
#         self.last_alert_time = 0
#         self.alert_cooldown = 5
#         self.sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"

#     def startCamera(self):
#         if not self.camera_active:
#             self.capture = cv2.VideoCapture(0)
#             self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#             self.timer = QTimer()
#             self.timer.timeout.connect(self.updateFrame)
#             self.timer.start(30)

#             self.camera_active = True
#             self.ui.actionCamera_ON.setText("Cam: OFF")
#         else:
#             self.timer.stop()
#             self.capture.release()
#             self.ui.streaming.clear()
#             self.ui.actionCamera_ON.setText("Cam: ON")
#             self.camera_active = False

#     def calculate_angle(self, a, b, c):
#         a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
#         ba, bc = a - b, c - b
#         if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
#             return np.nan
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

#     def play_alert_sound(self):
#         threading.Thread(target=playsound, args=(self.sound_file,), daemon=True).start()

#     def updateFrame(self):
#         ret, frame = self.capture.read()
#         if not ret:
#             return

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.pose.process(rgb_frame)

#         status = "Calibrating..."

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             h, w, _ = frame.shape

#             # Get important landmarks
#             left_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
#                                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y], [w, h]).astype(int))
#             right_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
#                                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y], [w, h]).astype(int))
#             left_ear = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x,
#                                           landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y], [w, h]).astype(int))
#             nose = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.NOSE].x,
#                                       landmarks[self.mp_pose.PoseLandmark.NOSE].y], [w, h]).astype(int))
#             midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
#                         (left_shoulder[1] + right_shoulder[1]) // 2)

#             shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
#             neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
#             nose_to_midpoint_distance = np.linalg.norm(np.array(nose) - np.array(midpoint))

#             if not self.is_calibrated and self.calibration_frames < 30:
#                 self.calibration_shoulder_angles.append(shoulder_angle)
#                 self.calibration_neck_angles.append(neck_angle)
#                 self.calibration_nose_distance.append(nose_to_midpoint_distance)
#                 self.calibration_frames += 1
#             elif not self.is_calibrated:
#                 self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 5
#                 self.neck_threshold = np.mean(self.calibration_neck_angles) - 5
#                 self.nose_threshold = np.mean(self.calibration_nose_distance) - 10
#                 self.is_calibrated = True

#             if self.is_calibrated:
#                 current_time = time.time()
#                 if (shoulder_angle < self.shoulder_threshold or
#                     neck_angle < self.neck_threshold or
#                     nose_to_midpoint_distance < self.nose_threshold):
#                     status = "Poor Posture"
#                     color = (0, 0, 255)
#                     if current_time - self.last_alert_time > self.alert_cooldown:
#                         self.play_alert_sound()
#                         self.last_alert_time = current_time
#                 else:
#                     status = "Good Posture"
#                     color = (0, 255, 0)

#                 for pt in [left_shoulder, right_shoulder, left_ear, nose, midpoint]:
#                     cv2.circle(frame, pt, 8, color, -1)

#                 # cv2.putText(frame, f"{status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                 # cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}/{self.shoulder_threshold:.1f}", (10, 60),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#                 # cv2.putText(frame, f"Neck: {neck_angle:.1f}/{self.neck_threshold:.1f}", (10, 85),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#                 # cv2.putText(frame, f"Nose: {nose_to_midpoint_distance:.1f}/{self.nose_threshold:.1f}", (10, 110),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

#             self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

#         # Convert to QImage and show
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

#         scaled_img = qt_img.scaled(self.ui.streaming.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
#                                    Qt.TransformationMode.SmoothTransformation)
#         label_size = self.ui.streaming.size()
#         rect = QRect(
#             (scaled_img.width() - label_size.width()) // 2,
#             (scaled_img.height() - label_size.height()) // 2,
#             label_size.width(),
#             label_size.height()
#         )
#         cropped_img = scaled_img.copy(rect)
#         self.ui.streaming.setPixmap(QPixmap.fromImage(cropped_img))

#     def closeEvent(self, event):
#         if hasattr(self, 'capture') and self.capture is not None:
#             self.timer.stop()
#             self.capture.release()
#             self.capture = None
#         event.accept()

################################################################################################################################
# '''
# Integrating mediapipe with PyQt
# Set QLabel as output
# '''
# from PyQt6.QtCore import *
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import *

# import cv2
# import numpy as np
# import time
# import threading
# import mediapipe as mp
# from playsound import playsound

# from FIX_display import Ui_MainWindow

# class MainController(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)

#         self.ui.streaming.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.ui.streaming.setScaledContents(True)
#         self.ui.actionCamera_ON.triggered.connect(self.startCamera)

#         self.camera_active = False
#         self.start_time = None #Representasi waktu


#         # Posture Detection Initialization
#         self.mp_pose = mp.solutions.pose
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#         self.is_calibrated = False
#         self.calibration_frames = 0
#         self.calibration_shoulder_angles = []
#         self.calibration_neck_angles = []
#         self.calibration_nose_distance = []
#         self.shoulder_threshold = 0
#         self.neck_threshold = 0
#         self.nose_threshold = 0
#         self.last_alert_time = 0
#         self.alert_cooldown = 5
#         self.sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"

#     def startCamera(self):
#         if not self.camera_active:
#             self.capture = cv2.VideoCapture(0)
#             self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#             self.timer = QTimer()
#             self.timer.timeout.connect(self.updateFrame)
#             self.timer.start(30)

#             self.start_time = time.time()  # Mulai hitung waktu
#             self.camera_active = True
#             self.ui.actionCamera_ON.setText("Cam: OFF")
#         else:
#             self.timer.stop()
#             self.capture.release()
#             self.ui.streaming.clear()
#             self.ui.status.clear()
#             self.ui.time.clear()
#             self.ui.degree.clear()
#             self.ui.actionCamera_ON.setText("Cam: ON")
#             self.camera_active = False


#     def calculate_angle(self, a, b, c):
#         a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
#         ba, bc = a - b, c - b
#         if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
#             return np.nan
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

#     def play_alert_sound(self):
#         threading.Thread(target=playsound, args=(self.sound_file,), daemon=True).start()

#     def updateFrame(self):
#         ret, frame = self.capture.read()
#         if not ret:
#             return

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.pose.process(rgb_frame)

#         status = "Calibrating..."

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             h, w, _ = frame.shape

#             # Get important landmarks
#             left_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
#                                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y], [w, h]).astype(int))
#             right_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
#                                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y], [w, h]).astype(int))
#             left_ear = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x,
#                                           landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y], [w, h]).astype(int))
#             nose = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.NOSE].x,
#                                       landmarks[self.mp_pose.PoseLandmark.NOSE].y], [w, h]).astype(int))
#             midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
#                         (left_shoulder[1] + right_shoulder[1]) // 2)

#             shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
#             neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
#             nose_to_midpoint_distance = np.linalg.norm(np.array(nose) - np.array(midpoint))

#             if not self.is_calibrated and self.calibration_frames < 30:
#                 self.calibration_shoulder_angles.append(shoulder_angle)
#                 self.calibration_neck_angles.append(neck_angle)
#                 self.calibration_nose_distance.append(nose_to_midpoint_distance)
#                 self.calibration_frames += 1
#             elif not self.is_calibrated:
#                 self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 5
#                 self.neck_threshold = np.mean(self.calibration_neck_angles) - 5
#                 self.nose_threshold = np.mean(self.calibration_nose_distance) - 10
#                 self.is_calibrated = True

#             if self.is_calibrated:
#                 current_time = time.time()
#                 if (shoulder_angle < self.shoulder_threshold or
#                     neck_angle < self.neck_threshold or
#                     nose_to_midpoint_distance < self.nose_threshold):
#                     status = "Poor Posture"
#                     color = (0, 0, 255)
#                     if current_time - self.last_alert_time > self.alert_cooldown:
#                         self.play_alert_sound()
#                         self.last_alert_time = current_time
#                 else:
#                     status = "Good Posture"
#                     color = (0, 255, 0)

#                 for pt in [left_shoulder, right_shoulder, left_ear, nose, midpoint]:
#                     cv2.circle(frame, pt, 8, color, -1)

#                 # Update QLabel status
#                 self.ui.status.setText(f"{status}")

#                 # Update QLabel Time
#                 elapsed_time = int(time.time() - self.start_time)
#                 self.ui.time.setText(f"Elapsed Time: {elapsed_time} s")
                
#                 # Update QLabel Degree
#                 self.ui.degree.setText(
#                     f"Shoulder: {shoulder_angle:.1f}° (Threshold: {self.shoulder_threshold:.1f}°)\n"
#                     f"Neck: {neck_angle:.1f}° (Threshold: {self.neck_threshold:.1f}°)\n"
#                     f"Nose Dist: {nose_to_midpoint_distance:.1f} (Threshold: {self.nose_threshold:.1f})"
#                 )


#             self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

#         # Convert to QImage and show
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

#         scaled_img = qt_img.scaled(self.ui.streaming.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
#                                    Qt.TransformationMode.SmoothTransformation)
#         label_size = self.ui.streaming.size()
#         rect = QRect(
#             (scaled_img.width() - label_size.width()) // 2,
#             (scaled_img.height() - label_size.height()) // 2,
#             label_size.width(),
#             label_size.height()
#         )
#         cropped_img = scaled_img.copy(rect)
#         self.ui.streaming.setPixmap(QPixmap.fromImage(cropped_img))

#     def closeEvent(self, event):
#         if hasattr(self, 'capture') and self.capture is not None:
#             self.timer.stop()
#             self.capture.release()
#             self.capture = None
#         event.accept()

##################################################################################################
# '''
# Integrating mediapipe with PyQt
# Integrating mediapipe with PyQt
# Set QLabel as output
# Make function for QAction: Mute & PopUp
# '''
# from PyQt6.QtCore import *
# from PyQt6.QtGui import *
# from PyQt6.QtWidgets import *

# import cv2
# import numpy as np
# import time
# import threading
# import mediapipe as mp
# from playsound import playsound

# from FIX_display import Ui_MainWindow

# class MainController(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)

#         self.ui.streaming.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.ui.streaming.setScaledContents(True)
#         self.ui.actionCamera_ON.triggered.connect(self.startCamera)
#         self.ui.actionMute.triggered.connect(self.Mute)
#         self.ui.actionPop_up_Mode.triggered.connect(self.PopUp)


#         self.camera_active = False
#         self.start_time = None #Representasi waktu

#         self.is_muted = False #Flag mode mute
#         self.popup_mode = False #Flag mode pop up

#         # Posture Detection Initialization
#         self.mp_pose = mp.solutions.pose
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#         self.is_calibrated = False
#         self.calibration_frames = 0
#         self.calibration_shoulder_angles = []
#         self.calibration_neck_angles = []
#         self.calibration_nose_distance = []
#         self.shoulder_threshold = 0
#         self.neck_threshold = 0
#         self.nose_threshold = 0
#         self.last_alert_time = 0
#         self.alert_cooldown = 5
#         self.sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"

#     def startCamera(self):
#         if not self.camera_active:
#             self.capture = cv2.VideoCapture(0)
#             self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#             self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#             self.timer = QTimer()
#             self.timer.timeout.connect(self.updateFrame)
#             self.timer.start(30)

#             self.start_time = time.time()  # Mulai hitung waktu
#             self.camera_active = True
#             self.ui.actionCamera_ON.setText("Cam: OFF")
#         else:
#             self.timer.stop()
#             self.capture.release()
#             self.ui.streaming.clear()
#             self.ui.status.clear()
#             self.ui.time.clear()
#             self.ui.degree.clear()
#             self.ui.actionCamera_ON.setText("Cam: ON")
#             self.camera_active = False


#     def calculate_angle(self, a, b, c):
#         a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
#         ba, bc = a - b, c - b
#         if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
#             return np.nan
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

#     def play_alert_sound(self):
#         if not self.is_muted:
#             threading.Thread(target=playsound, args=(self.sound_file,), daemon=True).start()

#     def updateFrame(self):
#         ret, frame = self.capture.read()
#         if not ret:
#             return

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.pose.process(rgb_frame)

#         status = "Calibrating..."

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             h, w, _ = frame.shape

#             # Get important landmarks
#             left_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
#                                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y], [w, h]).astype(int))
#             right_shoulder = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
#                                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y], [w, h]).astype(int))
#             left_ear = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x,
#                                           landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y], [w, h]).astype(int))
#             nose = tuple(np.multiply([landmarks[self.mp_pose.PoseLandmark.NOSE].x,
#                                       landmarks[self.mp_pose.PoseLandmark.NOSE].y], [w, h]).astype(int))
#             midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
#                         (left_shoulder[1] + right_shoulder[1]) // 2)

#             shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
#             neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
#             nose_to_midpoint_distance = np.linalg.norm(np.array(nose) - np.array(midpoint))

#             if not self.is_calibrated and self.calibration_frames < 30:
#                 self.calibration_shoulder_angles.append(shoulder_angle)
#                 self.calibration_neck_angles.append(neck_angle)
#                 self.calibration_nose_distance.append(nose_to_midpoint_distance)
#                 self.calibration_frames += 1
#             elif not self.is_calibrated:
#                 self.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 5
#                 self.neck_threshold = np.mean(self.calibration_neck_angles) - 5
#                 self.nose_threshold = np.mean(self.calibration_nose_distance) - 10
#                 self.is_calibrated = True

#             if self.is_calibrated:
#                 current_time = time.time()
#                 if (shoulder_angle < self.shoulder_threshold or
#                     neck_angle < self.neck_threshold or
#                     nose_to_midpoint_distance < self.nose_threshold):
#                     status = "Poor Posture"
#                     color = (0, 0, 255)
#                     if current_time - self.last_alert_time > self.alert_cooldown:
#                         self.play_alert_sound()
#                         self.last_alert_time = current_time
#                         if self.popup_mode:
#                             QMessageBox.warning(self, "Posture Alert", "Warning: Poor posture detected!", QMessageBox.StandardButton.Ok)

#                 else:
#                     status = "Good Posture"
#                     color = (0, 255, 0)

#                 for pt in [left_shoulder, right_shoulder, left_ear, nose, midpoint]:
#                     cv2.circle(frame, pt, 8, color, -1)

#                 # Update QLabel status
#                 self.ui.status.setText(f"{status}")

#                 # Update QLabel Time
#                 elapsed_time = int(time.time() - self.start_time)
#                 self.ui.time.setText(f"Elapsed Time: {elapsed_time} s")
                
#                 # Update QLabel Degree
#                 self.ui.degree.setText(
#                     f"Shoulder: {shoulder_angle:.1f}° (Threshold: {self.shoulder_threshold:.1f}°)\n"
#                     f"Neck: {neck_angle:.1f}° (Threshold: {self.neck_threshold:.1f}°)\n"
#                     f"Nose Dist: {nose_to_midpoint_distance:.1f} (Threshold: {self.nose_threshold:.1f})"
#                 )


#             self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

#         # Convert to QImage and show
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = frame.shape
#         bytes_per_line = ch * w
#         qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

#         scaled_img = qt_img.scaled(self.ui.streaming.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
#                                    Qt.TransformationMode.SmoothTransformation)
#         label_size = self.ui.streaming.size()
#         rect = QRect(
#             (scaled_img.width() - label_size.width()) // 2,
#             (scaled_img.height() - label_size.height()) // 2,
#             label_size.width(),
#             label_size.height()
#         )
#         cropped_img = scaled_img.copy(rect)
#         self.ui.streaming.setPixmap(QPixmap.fromImage(cropped_img))

#     def Mute(self):
#         self.is_muted = not self.is_muted
#         if self.is_muted:
#             self.ui.actionMute.setText("Unmute")
#             self.ui.statusbar.showMessage("Alert sound muted", 3000)
#         else:
#             self.ui.actionMute.setText("Mute")
#             self.ui.statusbar.showMessage("Alert sound unmuted", 3000)

#     def PopUp(self):
#         self.popup_mode = not self.popup_mode
#         if self.popup_mode:
#             self.ui.actionPop_up_Mode.setText("Pop-up: ON")
#             self.ui.statusbar.showMessage("Pop-up mode enabled", 3000)
#         else:
#             self.ui.actionPop_up_Mode.setText("Pop-up: OFF")
#             self.ui.statusbar.showMessage("Pop-up mode disabled", 3000)

            


#     def closeEvent(self, event):
#         if hasattr(self, 'capture') and self.capture is not None:
#             self.timer.stop()
#             self.capture.release()
#             self.capture = None
#         event.accept()


'''
Integrating mediapipe with PyQt
Set QLabel as output
Make function for QAction: Mute & PopUp
Make function for QAction: Select Notification & About
'''
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import cv2
import numpy as np
import time
import threading
import mediapipe as mp
from playsound import playsound

from FIX_display import Ui_MainWindow

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.streaming.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui.streaming.setScaledContents(True)
        self.ui.actionCamera_ON.triggered.connect(self.startCamera)
        self.ui.actionMute.triggered.connect(self.Mute)
        self.ui.actionPop_up_Mode.triggered.connect(self.PopUp)
        self.ui.actionSelect_Notification.triggered.connect(self.selectNotification)
        self.ui.actionAbout.triggered.connect(self.About)

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
        else:
            self.timer.stop()
            self.capture.release()
            self.ui.streaming.clear()
            self.ui.status.clear()
            self.ui.time.clear()
            self.ui.degree.clear()
            self.ui.actionCamera_ON.setText("Cam: ON")
            self.camera_active = False


    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
        ba, bc = a - b, c - b
        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return np.nan
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def play_alert_sound(self):
        if not self.is_muted:
            threading.Thread(target=playsound, args=(self.sound_file,), daemon=True).start()

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
                if (shoulder_angle < self.shoulder_threshold or
                    neck_angle < self.neck_threshold or
                    nose_to_midpoint_distance < self.nose_threshold):
                    status = "Poor Posture"
                    color = (0, 0, 255)
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        self.play_alert_sound()
                        self.last_alert_time = current_time
                        if self.popup_mode:
                            QMessageBox.warning(self, "Posture Alert", "Warning: Poor posture detected!", QMessageBox.StandardButton.Ok)

                else:
                    status = "Good Posture"
                    color = (0, 255, 0)

                for pt in [left_shoulder, right_shoulder, left_ear, nose, midpoint]:
                    cv2.circle(frame, pt, 8, color, -1)

                # Update QLabel status
                self.ui.status.setText(f"{status}")

                # Update QLabel Time
                elapsed_time = int(time.time() - self.start_time)
                self.ui.time.setText(f"Elapsed Time: {elapsed_time} s")
                
                # Update QLabel Degree
                self.ui.degree.setText(
                    f"Shoulder: {shoulder_angle:.1f}° (Threshold: {self.shoulder_threshold:.1f}°)\n"
                    f"Neck: {neck_angle:.1f}° (Threshold: {self.neck_threshold:.1f}°)\n"
                    f"Nose Dist: {nose_to_midpoint_distance:.1f} (Threshold: {self.nose_threshold:.1f})"
                )


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
        options = ["Sound Only", "Pop-up Only", "Sound + Pop-up", "None"]
        choice, ok = QInputDialog.getItem(self, "Select Notification Mode",
                                        "Choose how you want to receive posture alerts:",
                                        options, 0, False)
        if ok and choice:
            if choice == "Sound Only":
                self.is_muted = False
                self.popup_mode = False
                self.ui.actionMute.setText("Unmute")
                self.ui.actionPop_up_Mode.setText("Pop-up: OFF")
                self.ui.statusbar.showMessage("Notification mode: Sound Only", 3000)
            elif choice == "Pop-up Only":
                self.is_muted = True
                self.popup_mode = True
                self.ui.actionMute.setText("Mute")
                self.ui.actionPop_up_Mode.setText("Pop-up: ON")
                self.ui.statusbar.showMessage("Notification mode: Pop-up Only", 3000)
            elif choice == "Sound + Pop-up":
                self.is_muted = False
                self.popup_mode = True
                self.ui.actionMute.setText("Unmute")
                self.ui.actionPop_up_Mode.setText("Pop-up: ON")
                self.ui.statusbar.showMessage("Notification mode: Sound + Pop-up", 3000)
            elif choice == "None":
                self.is_muted = True
                self.popup_mode = False
                self.ui.actionMute.setText("Mute")
                self.ui.actionPop_up_Mode.setText("Pop-up: OFF")
                self.ui.statusbar.showMessage("Notification mode: None", 3000)
    
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




        
        
    def closeEvent(self, event):
        if hasattr(self, 'capture') and self.capture is not None:
            self.timer.stop()
            self.capture.release()
            self.capture = None
        event.accept()

