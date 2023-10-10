import cv2
import mediapipe as mp
import math
import time
from playsound import playsound

def compute_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_open_palm(landmarks):
    # Check if fingers are open
    if any(landmarks[finger_tip].y >= landmarks[finger_mcp].y for finger_tip, finger_mcp in [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_MCP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)]):
        return False

    # Checking if palm is approximately facing the camera using z coordinates
    if landmarks[mp_hands.HandLandmark.WRIST].z - landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z < -0.05:
        return False

    return True


# Introducing a cooldown mechanism
COOLDOWN_FRAMES = 30
cooldown_counter = 0

# Initialize mediapipe face mesh utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

# Initialize a countdown variable and a countdown active flag
# COUNTDOWN_TIME = 4  # 3 seconds
# frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))
# countdown = 1
# countdown_active = False
# Initialize countdown variables
countdown_end_time = None
countdown_active = False
COUNTDOWN_TIME = 4  # 3 seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hands
    hand_results = hands.process(rgb_frame)

    # Process frame for hands
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks:
        for facial_landmarks in face_results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            # Get landmarks for eye blink detection
            left_eye_top = (int(facial_landmarks.landmark[159].x * iw), int(facial_landmarks.landmark[159].y * ih))
            left_eye_bottom = (int(facial_landmarks.landmark[145].x * iw), int(facial_landmarks.landmark[145].y * ih))
            right_eye_top = (int(facial_landmarks.landmark[386].x * iw), int(facial_landmarks.landmark[386].y * ih))
            right_eye_bottom = (int(facial_landmarks.landmark[374].x * iw), int(facial_landmarks.landmark[374].y * ih))
            
            left_eye = (int(facial_landmarks.landmark[33].x * iw), int(facial_landmarks.landmark[33].y * ih))
            right_eye = (int(facial_landmarks.landmark[263].x * iw), int(facial_landmarks.landmark[263].y * ih))
            angle = compute_angle(left_eye, right_eye)

            # Blink detection
            left_eye_ear = euclidean_distance(left_eye_top, left_eye_bottom)
            right_eye_ear = euclidean_distance(right_eye_top, right_eye_bottom)
            eye_ear_avg = (left_eye_ear + right_eye_ear) / 2.0

            # Threshold for EAR (depends on the camera resolution and distance from the camera)
            EAR_THRESHOLD = 5  # Adjust this based on your observations

            if eye_ear_avg < EAR_THRESHOLD:
                cv2.putText(frame, "BLINK DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check for open palm
    
    if hand_results.multi_hand_landmarks and not countdown_active:
        for landmarks in hand_results.multi_hand_landmarks:
            if is_open_palm(landmarks.landmark):
                countdown_active = True
                countdown_end_time = time.time() + COUNTDOWN_TIME

    # Display countdown if active
    if countdown_active:
        seconds_remaining = int(countdown_end_time - time.time())

        if seconds_remaining > 0:
            # playsound('./stop_beep.mp3')
            cv2.putText(frame, str(seconds_remaining), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            countdown_active = False
            
            if facial_landmarks:  # ensure you've detected facial landmarks before attempting to take photo
                # Constants for US passport photo
                PASSPORT_WIDTH = 800  # in pixels
                PASSPORT_HEIGHT = 800  # in pixels

                # Convert inches to pixels for the given photo size
                HEAD_HEIGHT_MIN = int(PASSPORT_HEIGHT * 25 / 50)  # 1 inch in pixels
                HEAD_HEIGHT_MAX = int(PASSPORT_HEIGHT * 35 / 50)  # 1.375 inches in pixels
                EYES_HEIGHT_FROM_BOTTOM_MIN = int(PASSPORT_HEIGHT * 28 / 50)  # 1 1/8 inches from bottom
                EYES_HEIGHT_FROM_BOTTOM_MAX = int(PASSPORT_HEIGHT * 35 / 50)  # 1.375 inches from bottom

                # Landmarks for forehead and chin
                chin = (int(facial_landmarks.landmark[152].x * iw), int(facial_landmarks.landmark[152].y * ih))
                forehead = (int(facial_landmarks.landmark[10].x * iw), int(facial_landmarks.landmark[10].y * ih))

                # Rotate the frame
                center_x = (left_eye[0] + right_eye[0]) // 2
                center_y = (left_eye[1] + right_eye[1]) // 2
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
                rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

                # Calculate actual height of the head in the photo
                actual_face_height = chin[1] - forehead[1]

                # Calculate scaling factor based on desired head height
                desired_face_height = (HEAD_HEIGHT_MIN + HEAD_HEIGHT_MAX) / 2
                scale_factor = (desired_face_height / actual_face_height) * 0.75

                # Calculate new eye positions post-resize
                left_eye_resized = (int(left_eye[0] * scale_factor), int(left_eye[1] * scale_factor))
                right_eye_resized = (int(right_eye[0] * scale_factor), int(right_eye[1] * scale_factor))

                # Get the average y-coordinate of the eyes
                eyes_avg_y = (left_eye_resized[1] + right_eye_resized[1]) / 2

                # Determine the y coordinate for the bottom of the cropped image, based on desired eye position from bottom
                bottom_y = eyes_avg_y + (EYES_HEIGHT_FROM_BOTTOM_MIN + EYES_HEIGHT_FROM_BOTTOM_MAX) / 2

                # Calculate the position to crop
                x_center = int((left_eye[0] + right_eye[0]) * scale_factor / 2)
                y_center = int(bottom_y - PASSPORT_HEIGHT)

                resized_frame = cv2.resize(rotated_frame, (int(rotated_frame.shape[1] * scale_factor), int(rotated_frame.shape[0] * scale_factor)))

                # Ensure cropping coordinates are valid
                top_left_y = max(0, y_center)
                bottom_right_y = min(resized_frame.shape[0], y_center + PASSPORT_HEIGHT)
                top_left_x = max(0, x_center - PASSPORT_WIDTH // 2)
                bottom_right_x = min(resized_frame.shape[1], x_center + PASSPORT_WIDTH // 2)


                # Check if the resulting crop would have valid dimensions
                if bottom_right_y - top_left_y > 0 and bottom_right_x - top_left_x > 0:
                    
                    playsound('./camera.mp3')

                    # Crop the image
                    cropped_passport = resized_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                    cv2.imshow("Passport Photo", cropped_passport)

                    # Save the cropped image as a .jpg file
                    filename = "passport_photo.jpg"
                    cv2.imwrite(filename, cropped_passport)
                    print(f"Passport photo saved as {filename}")

                    cv2.waitKey(0)
                    cv2.destroyWindow("Passport Photo")
                else:
                    print("Face not detected properly.")
                    continue

    # Decrease cooldown counter
    if cooldown_counter > 0:
        cooldown_counter -= 1

    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
