import cv2
import mediapipe as mp
import math

def compute_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Initialize mediapipe face detection and drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Get landmarks
            landmarks = detection.location_data.relative_keypoints
            left_eye = (int(landmarks[0].x * iw), int(landmarks[0].y * ih))
            right_eye = (int(landmarks[1].x * iw), int(landmarks[1].y * ih))
            angle = compute_angle(left_eye, right_eye)

    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1)
    if key == ord('t'):
        if 'left_eye' in locals() and 'right_eye' in locals():
            # Constants for US passport photo
            PASSPORT_WIDTH = 600  # in pixels
            PASSPORT_HEIGHT = 600  # in pixels
            FACE_HEIGHT_RATIO = 0.75  # desired face height / photo height
            EYES_POSITION_RATIO = 1/3  # position from the bottom

            # Rotate the frame
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
            rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            # Calculate scale factor to get desired face height
            face_height = y + h - y
            desired_face_height = int(FACE_HEIGHT_RATIO * PASSPORT_HEIGHT)
            scale_factor = desired_face_height / face_height

            # Resize the rotated image
            resized_frame = cv2.resize(rotated_frame, (int(rotated_frame.shape[1] * scale_factor), int(rotated_frame.shape[0] * scale_factor)))

            # Calculate the position to crop
            x_center = int(x * scale_factor + w * scale_factor / 2)
            y_center = int(y * scale_factor + h * scale_factor / 2)

            top_left_y = int(y_center - (PASSPORT_HEIGHT * EYES_POSITION_RATIO))
            bottom_right_y = top_left_y + PASSPORT_HEIGHT
            top_left_x = x_center - (PASSPORT_WIDTH // 2)
            bottom_right_x = top_left_x + PASSPORT_WIDTH

            # Crop the image
            cropped_passport = resized_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            cv2.imshow("Passport Photo", cropped_passport)
            cv2.waitKey(0)
            cv2.destroyWindow("Passport Photo")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
