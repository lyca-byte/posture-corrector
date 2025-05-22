#Menambahkan parameter punggung lalu dihitung sudutnya berdasarka itu

import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import threading

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Utk gambar titik-titik sendi tubuh
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Variabel kalibrasi
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
calibration_hip_angles = []  # Variabel untuk kalibrasi sudut pinggul
shoulder_threshold = 0
neck_threshold = 0
hip_threshold = 0  # Threshold untuk pinggul
last_alert_time = 0
alert_cooldown = 5  # Waktu cooldown peringatan dalam detik
sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"  # Ganti dengan path file suara yang digunakan

def calculate_angle(a, b, c):
    """
    Menghitung sudut antara tiga titik (a, b, c). 
    Sudut dihitung dalam derajat.
    Digunakan untuk menghitung sudut bahu, leher, dan pinggul
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def draw_angle(image, pointA, pointB, pointC, angle, color=(255, 255, 255)):
    """
    Menggambar sudut pada gambar menggunakan tiga titik: point A, point B, dan point C.
    Menggambar garis antara tiga titik
    Menampilkan nilai sudut di samping titik tengah
    """
    cv2.line(image, pointA, pointB, color, 2)
    cv2.line(image, pointB, pointC, color, 2)
    
    # Menampilkan nilai sudut pada gambar
    cv2.putText(image, f"{int(angle)} deg", (pointB[0] + 10, pointB[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def play_alert_sound():
    playsound(sound_file)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # STEP 2: Pose Detection
        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]))

        # STEP 3: Angle Calculation
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))  # Sudut bahu
        neck_angle_left = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))  # Sudut leher kiri
        neck_angle_right = calculate_angle(right_ear, right_shoulder, (right_shoulder[0], 0))  # Sudut leher kanan
        hip_angle = calculate_angle(left_hip, right_hip, (right_hip[0], 0))  # Sudut pinggul

        # STEP 1: Calibration
        if not is_calibrated and calibration_frames < 30:  # Sistem mengumpulkan 30 frame pertama untuk menentukan sudut rata-rata postur normal
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append((neck_angle_left + neck_angle_right) / 2)  # Rata-rata sudut leher kiri dan kanan
            calibration_hip_angles.append(hip_angle)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 5
            neck_threshold = np.mean(calibration_neck_angles) - 5
            hip_threshold = np.mean(calibration_hip_angles) - 5
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}, Hip threshold: {hip_threshold:.1f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle_left, (0, 255, 0))
        draw_angle(frame, right_ear, right_shoulder, (right_shoulder[0], 0), neck_angle_right, (0, 255, 0))
        draw_angle(frame, left_hip, right_hip, (right_hip[0], 0), hip_angle, (0, 0, 255))

        # STEP 4: Feedback
        if is_calibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or (neck_angle_left < neck_threshold and neck_angle_right < neck_threshold) or hip_angle < hip_threshold:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Poor posture detected! Please sit up straight.")
                    threading.Thread(target=play_alert_sound).start()
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: L:{neck_angle_left:.1f}/{neck_threshold:.1f} R:{neck_angle_right:.1f}/{neck_threshold:.1f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Hip Angle: {hip_angle:.1f}/{hip_threshold:.1f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
