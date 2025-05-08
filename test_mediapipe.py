import cv2
import mediapipe as mp
import time
import numpy as np

# Khởi tạo tất cả các module cần thiết
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo từng module riêng biệt
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Tùy chỉnh màu sắc cho pose và hands
pose_landmark_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1)
pose_connection_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=1)

hand_landmark_spec = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=1)
hand_connection_spec = mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)

# 33 - 263: khóe mắt trái
# 61 - 291: miệng
# 468 - 473: con ngươi

face_keypoints = [1, 468, 61, 199, 473, 291]
face_point_color = (255, 0, 255)  # Màu tím
face_point_size = 5

# Mở camera
cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(frame_rgb)

    pose_results = pose.process(frame_rgb)

    hands_results = hands.process(frame_rgb)

    ih, iw, _ = frame.shape

    final_frame = frame.copy()
    overlay = np.zeros_like(frame)

    # PHẦN POSE: Vẽ pose landmarks và kết nối lên overlay
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            overlay,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            pose_landmark_spec,
            pose_connection_spec
        )

    # PHẦN HANDS: Vẽ hand landmarks và kết nối lên overlay
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                overlay,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_landmark_spec,
                hand_connection_spec
            )

    # XÓA PHẦN MẶT: Tạo mask để xóa phần vẽ trên khuôn mặt nếu có face
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Tạo convex hull từ các điểm mặt để xác định vùng mặt
            face_points = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * iw), int(lm.y * ih)
                face_points.append((x, y))

            # Chỉ lấy các điểm phía ngoài để tạo hull
            if len(face_points) > 3:  # Cần ít nhất 4 điểm để tạo hull
                face_hull = cv2.convexHull(np.array(face_points))

                # Tạo mask cho khuôn mặt
                face_mask = np.zeros((ih, iw), dtype=np.uint8)
                cv2.fillConvexPoly(face_mask, face_hull, 255)

                for c in range(3):
                    overlay[:, :, c] = cv2.bitwise_and(
                        overlay[:, :, c],
                        overlay[:, :, c],
                        mask=cv2.bitwise_not(face_mask)
                    )

    # PHẦN MẶT: Vẽ các điểm mặt quan trọng lên final_frame
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx in face_keypoints:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(final_frame, (x, y), face_point_size, face_point_color, -1)


    final_frame = cv2.add(final_frame, overlay)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
    prev_time = curr_time

    cv2.putText(final_frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Points + Pose + Hands (No Face Mesh)', final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()