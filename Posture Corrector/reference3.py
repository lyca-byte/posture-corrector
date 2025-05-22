'''
-Ditambah prameter landmark nose sebagai parameter untuk menentukan apakah kamera mendeteksi postur bungkuk.
-Dari landmark nose, tarik garis ke bagian tengah landmark bahu kanan dan kiri (disebut sebagai x).
-Jika value jarak landmark nose lebih kecil nilainya dari batas threshold (jarak antara landmark nose terlalu dekat dengan titik x), maka dipastikan
orang tersebut bungkuk.
-Jika value jarak landmark nose lebih besar nilainya dari batas threshold (jarak antara landmark nose tidak dekat dengan titik x), maka dapat
orang tersebut tidak dalam posisi bungkuk.
-Hal-hal ini perlu pertimbangan dan kalibrasi.
'''

# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# from playsound import playsound
# import threading

# # Initialize MediaPipe Pose and webcam
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# '''
# MediaPipe Pose is a ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks and background segmentation 
# mask on the whole body from RGB video frames utilizing our BlazePose research that also powers the ML Kit Pose Detection API.
# Current state-of-the-art approaches rely primarily on powerful desktop environments for inference, whereas our method achieves 
# real-time performance on most modern mobile phones, desktops/laptops, in python and even on the web.
# Configurations : 
# x.Pose(static_image_mode, model_complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, min_detection_confidence, 
# min_tracking_confidence)
# *Documentation in GitHub :/mediapipe/docs/
# '''
# cap = cv2.VideoCapture(0)

# # Variabel kalibrasi
# is_calibrated = False
# calibration_frames = 0
# calibration_shoulder_angles = []
# calibration_neck_angles = []
# calibration_nose_distance = []
# shoulder_threshold = 0
# neck_threshold = 0
# nose_threshold = 0
# last_alert_time = 0
# alert_cooldown = 5  # Waktu cooldown peringatan dalam detik
# sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"  # Ganti dengan path file suara yang digunakan

# # def calculate_angle(a, b, c):
# #     a = np.array(a)
# #     b = np.array(b)
# #     c = np.array(c)
# #     ba = a - b
# #     bc = c - b
# #     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
# #     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
# #     return np.degrees(angle)

# def calculate_angle(a, b, c):
#     a = np.array(a, dtype=np.float32)
#     b = np.array(b, dtype=np.float32)
#     c = np.array(c, dtype=np.float32)
#     ba = a - b
#     bc = c - b
#     norm_ba = np.linalg.norm(ba)
#     norm_bc = np.linalg.norm(bc)
#     if norm_ba == 0 or norm_bc == 0:
#         return np.nan  # Tidak valid
#     cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#     return np.degrees(angle)


# # def draw_angle(image, pointA, pointB, pointC, angle, color=(255, 255, 255)):
# #     cv2.line(image, pointA, pointB, color, 2)
# #     cv2.line(image, pointB, pointC, color, 2)
# #     cv2.putText(image, f"{int(angle)} deg", (pointB[0] + 10, pointB[1] - 10),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# def draw_angle(image, pointA, pointB, pointC, angle, color=(255, 255, 255)):
#     if np.isnan(angle):
#         return  # Jangan gambar kalau sudut tidak valid
#     cv2.line(image, pointA, pointB, color, 2)
#     cv2.line(image, pointB, pointC, color, 2)
#     cv2.putText(image, f"{int(angle)} deg", (pointB[0] + 10, pointB[1] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


# def play_alert_sound():
#     playsound(sound_file)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark

#         # Ambil titik tubuh
#         left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
#                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
#         right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
#                           int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
#         left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
#                     int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
#         right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
#                      int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))
#         nose = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1]),
#                 int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]))

#         # Midpoint antara bahu kanan dan kiri
#         midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
#                     (left_shoulder[1] + right_shoulder[1]) // 2)

#         # Hitung sudut
#         shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
#         neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
#         nose_to_midpoint_distance = abs(nose[1] - midpoint[1])  # [NEW]

#         # Kalibrasi awal
#         if not is_calibrated and calibration_frames < 30:
#             calibration_shoulder_angles.append(shoulder_angle)
#             calibration_neck_angles.append(neck_angle)
#             calibration_nose_distance.append(nose_to_midpoint_distance)
#             calibration_frames += 1
#             cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#         elif not is_calibrated:
#             shoulder_threshold = np.mean(calibration_shoulder_angles) - 5
#             neck_threshold = np.mean(calibration_neck_angles) - 5
#             nose_threshold = np.mean(calibration_nose_distance) + 10
#             is_calibrated = True
#             print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")

#         # Gambar landmark dan garis sudut
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
#         draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
#         cv2.line(frame, nose, midpoint, (0, 255, 255), 2)

#         # Feedback
#         if is_calibrated:
#             current_time = time.time()
#             if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold or nose_to_midpoint_distance > nose_threshold:
#                 status = "Poor Posture"
#                 color = (0, 0, 255)
#                 if current_time - last_alert_time > alert_cooldown:
#                     print("Poor posture detected! Please sit up straight.")
#                     threading.Thread(target=play_alert_sound).start()
#                     last_alert_time = current_time
#             else:
#                 status = "Good Posture"
#                 color = (0, 255, 0)

#             cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#             cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#             cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
#             cv2.putText(frame, f"Nose Dist. : {nose_to_midpoint_distance:.1f}/{nose_threshold:.1f}", (10, 120),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

#     # Tampilkan hasil
#     cv2.imshow('Posture Corrector', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


##########################################################################################################################
#Logika landmark nose dengan midpoint bahu kanan dan kiri, kalau distancenya besar berarti posenya salah

# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# from playsound import playsound
# import threading

# # Initialize MediaPipe Pose and webcam
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# cap = cv2.VideoCapture(0)

# # Kalibrasi & konfigurasi
# is_calibrated = False
# calibration_frames = 0
# calibration_shoulder_angles = []
# calibration_neck_angles = []
# calibration_nose_distance = []
# shoulder_threshold = 0
# neck_threshold = 0
# nose_threshold = 0
# last_alert_time = 0
# alert_cooldown = 5
# sound_file = "C:/Users/ASUS/Documents/AI/pn.mp3"

# # Fungsi perhitungan sudut
# def calculate_angle(a, b, c):
#     a = np.array(a, dtype=np.float32)
#     b = np.array(b, dtype=np.float32)
#     c = np.array(c, dtype=np.float32)
#     ba = a - b
#     bc = c - b
#     norm_ba = np.linalg.norm(ba)
#     norm_bc = np.linalg.norm(bc)
#     if norm_ba == 0 or norm_bc == 0:
#         return np.nan
#     cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#     return np.degrees(angle)

# # Fungsi menggambar sudut
# def draw_angle(image, pointA, pointB, pointC, angle, color=(255, 255, 255)):
#     if np.isnan(angle):
#         return
#     cv2.line(image, pointA, pointB, color, 2)
#     cv2.line(image, pointB, pointC, color, 2)
#     cv2.putText(image, f"{int(angle)} deg", (pointB[0] + 10, pointB[1] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# # Fungsi menggambar jarak hidung ke midpoint bahu
# def draw_nose_distance(image, nose, midpoint, distance, color=(0, 255, 255)):
#     if np.isnan(distance):
#         return
#     cv2.line(image, nose, midpoint, color, 2)
#     mid_text_x = (nose[0] + midpoint[0]) // 2
#     mid_text_y = (nose[1] + midpoint[1]) // 2 - 10
#     cv2.putText(image, f"{int(distance)} px", (mid_text_x, mid_text_y),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# # Fungsi main loop
# def play_alert_sound():
#     playsound(sound_file)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb_frame)

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark

#         # Landmark penting
#         left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
#                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]))
#         right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
#                           int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]))
#         left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * frame.shape[1]),
#                     int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * frame.shape[0]))
#         nose = (int(landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1]),
#                 int(landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]))

#         midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
#                     (left_shoulder[1] + right_shoulder[1]) // 2)

#         shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
#         neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
#         nose_to_midpoint_distance = np.linalg.norm(np.array(nose) - np.array(midpoint))

#         # Kalibrasi awal
#         if not is_calibrated and calibration_frames < 30:
#             calibration_shoulder_angles.append(shoulder_angle)
#             calibration_neck_angles.append(neck_angle)
#             calibration_nose_distance.append(nose_to_midpoint_distance)
#             calibration_frames += 1
#             cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         elif not is_calibrated:
#             shoulder_threshold = np.mean(calibration_shoulder_angles) - 5
#             neck_threshold = np.mean(calibration_neck_angles) - 5
#             nose_threshold = np.mean(calibration_nose_distance) + 10
#             is_calibrated = True
#             print(f"Calibration done.\nShoulder: {shoulder_threshold:.1f}, Neck: {neck_threshold:.1f}, Nose: {nose_threshold:.1f}")

#         # Gambar
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
#         draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
#         draw_nose_distance(frame, nose, midpoint, nose_to_midpoint_distance, (0, 255, 255))

#         # Deteksi postur
#         if is_calibrated:
#             current_time = time.time()
#             if (shoulder_angle < shoulder_threshold or
#                 neck_angle < neck_threshold or
#                 nose_to_midpoint_distance > nose_threshold):
#                 status = "Poor Posture"
#                 color = (0, 0, 255)
#                 if current_time - last_alert_time > alert_cooldown:
#                     threading.Thread(target=play_alert_sound).start()
#                     print("Poor posture detected!")
#                     last_alert_time = current_time
#             else:
#                 status = "Good Posture"
#                 color = (0, 255, 0)

#             # Status teks
#             cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#             cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#             cv2.putText(frame, f"Neck: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 85),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#             cv2.putText(frame, f"Nose Dist: {nose_to_midpoint_distance:.1f}/{nose_threshold:.1f}", (10, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

#     # Tampilkan frame
#     cv2.imshow('Posture Corrector', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

###################################################################################################################################
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
    mid_text = ((nose[0] + midpoint[0]) // 2, (nose[1] + midpoint[1]) // 2 + 10) #Harusnya min 10
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
            nose_threshold = np.mean(calibration_nose_distance) - 10  # ✅ Perubahan
            is_calibrated = True
            print(f"Calibration done.\nShoulder: {shoulder_threshold:.1f}, Neck: {neck_threshold:.1f}, Nose: {nose_threshold:.1f}")

        # Visualisasi
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
        draw_nose_distance(frame, nose, midpoint, nose_to_midpoint_distance)

        # Deteksi Postur
        if is_calibrated:
            current_time = time.time()
            if (shoulder_angle < shoulder_threshold or
                neck_angle < neck_threshold or
                nose_to_midpoint_distance < nose_threshold):  # ✅ Perubahan
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



