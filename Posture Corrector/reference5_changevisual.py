'''FIX VERSION'''
#Logika landmark nose dengan midpoint bahu kanan dan kiri, kalau distancenya kecil berarti posenya membungkuk [FIXED]
import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import threading

# Inisialisasi MediaPipe dan Webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Variabel Kalibrasi dan Konfigurasi
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
calibration_nose_distance = []
shoulder_threshold = 0
neck_threshold = 0
nose_threshold = 0
last_alert_time = 0
alert_cooldown = 5
sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"

# Fungsi Hitung Sudut
def calculate_angle(a, b, c):
    a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
    ba, bc = a - b, c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return np.nan
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Fungsi Gambar Sudut
def draw_angle(image, pointA, pointB, pointC, angle, color=(255, 255, 255)):
    if np.isnan(angle): return
    cv2.line(image, pointA, pointB, color, 2)
    cv2.line(image, pointB, pointC, color, 2)
    cv2.putText(image, f"{int(angle)} deg", (pointB[0] + 10, pointB[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Fungsi Gambar Jarak Hidung ke Midpoint Bahu
def draw_nose_distance(image, nose, midpoint, distance, color=(0, 255, 255)):
    if np.isnan(distance): return
    cv2.line(image, nose, midpoint, color, 2)
    mid_text = ((nose[0] + midpoint[0]) // 2, (nose[1] + midpoint[1]) // 2 + 10) #Awalnya min 10 
    cv2.putText(image, f"{int(distance)} px", mid_text,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Fungsi Bunyi Alarm
def play_alert_sound():
    playsound(sound_file)

# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Landmark Penting
        left_shoulder = tuple(np.multiply([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                                           [frame.shape[1], frame.shape[0]]).astype(int))
        right_shoulder = tuple(np.multiply([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                                            [frame.shape[1], frame.shape[0]]).astype(int))
        left_ear = tuple(np.multiply([landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_EAR].y],
                                      [frame.shape[1], frame.shape[0]]).astype(int))
        nose = tuple(np.multiply([landmarks[mp_pose.PoseLandmark.NOSE].x,
                                  landmarks[mp_pose.PoseLandmark.NOSE].y],
                                  [frame.shape[1], frame.shape[0]]).astype(int))
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)

        # Perhitungan
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
        nose_to_midpoint_distance = np.linalg.norm(np.array(nose) - np.array(midpoint))

        # Kalibrasi Awal
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_nose_distance.append(nose_to_midpoint_distance)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 5
            neck_threshold = np.mean(calibration_neck_angles) - 5
            nose_threshold = np.mean(calibration_nose_distance) - 10  
            is_calibrated = True
            print(f"Calibration done.\nShoulder: {shoulder_threshold:.1f}, Neck: {neck_threshold:.1f}, Nose: {nose_threshold:.1f}")

        # Visualisasi
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        # draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
        # draw_nose_distance(frame, nose, midpoint, nose_to_midpoint_distance)

        # Gambar garis dan landmark pose default
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Tentukan warna berdasarkan postur
        color = (0, 255, 0) if (shoulder_angle >= shoulder_threshold and 
                                neck_angle >= neck_threshold and 
                                nose_to_midpoint_distance >= nose_threshold) else (0, 0, 255)

        # Sorot landmark penting dengan lingkaran
        for point in [left_shoulder, right_shoulder, left_ear, nose, midpoint]:
            cv2.circle(frame, point, 8, color, -1)

        # Gambar sudut dan jarak dengan warna dinamis
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, color)
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, color)
        draw_nose_distance(frame, nose, midpoint, nose_to_midpoint_distance, color)

        # Tambahkan overlay info
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 130), (0, 0, 0), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Tampilkan status dan data metrik
        icon = "✅" if color == (0, 255, 0) else "❌"
        cv2.putText(frame, f"{icon} {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f} / {shoulder_threshold:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Neck: {neck_angle:.1f} / {neck_threshold:.1f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Nose Dist: {nose_to_midpoint_distance:.1f} / {nose_threshold:.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


        # Deteksi Postur
        if is_calibrated:
            current_time = time.time()
            if (shoulder_angle < shoulder_threshold or
                neck_angle < neck_threshold or
                nose_to_midpoint_distance < nose_threshold): 
                status = "Poor Posture"
                color = (0, 0, 255)
                if current_time - last_alert_time > alert_cooldown:
                    threading.Thread(target=play_alert_sound).start()
                    print("Poor posture detected!")
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)

            # Tampilkan Status
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Neck: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Nose Dist: {nose_to_midpoint_distance:.1f}/{nose_threshold:.1f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Tampilkan Frame
    cv2.imshow('Posture Corrector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

