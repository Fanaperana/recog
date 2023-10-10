import cv2
import dlib
import math

def compute_angle(point1, point2):
    return math.degrees(math.atan2(point2.y - point1.y, point2.x - point1.x))

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    face_detected = False
    ANGLE = 0.0

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        left_eye = face_landmarks.part(36)
        right_eye = face_landmarks.part(45)
        ANGLE = compute_angle(left_eye, right_eye)
        face_detected = True

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)


    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1)
    if key == ord('t') and face_detected: 
        if 'left_eye' in locals() and 'right_eye' in locals():
            # Constants for US passport photo
            PASSPORT_WIDTH = 600  # in pixels
            PASSPORT_HEIGHT = 600  # in pixels
            FACE_HEIGHT_RATIO = 0.75  # desired face height / photo height
            EYES_POSITION_RATIO = 1/3  # position from the bottom

            # Rotate the frame
            center_x = (left_eye.x + right_eye.x) // 2
            center_y = (left_eye.y + right_eye.y) // 2
            M = cv2.getRotationMatrix2D((center_x, center_y), -ANGLE, 1)
            rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            # Calculate scale factor to get desired face height
            face_height = face.bottom() - face.top()
            desired_face_height = int(FACE_HEIGHT_RATIO * PASSPORT_HEIGHT)
            scale_factor = desired_face_height / face_height

            # Resize the rotated image
            resized_frame = cv2.resize(rotated_frame, (int(rotated_frame.shape[1] * scale_factor), int(rotated_frame.shape[0] * scale_factor)))

            # Calculate the position to crop
            x_center = int(face.left() * scale_factor + (face.width() * scale_factor) / 2)
            y_center = int(face.top() * scale_factor + (face.height() * scale_factor) / 2)

            top_left_y = int(y_center - (PASSPORT_HEIGHT * EYES_POSITION_RATIO))
            bottom_right_y = top_left_y + PASSPORT_HEIGHT
            top_left_x = x_center - (PASSPORT_WIDTH // 2)
            bottom_right_x = top_left_x + PASSPORT_WIDTH

            # Crop the image
            cropped_passport = resized_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            cv2.imshow("Passport Photo", cropped_passport)
            cv2.waitKey(0)
            cap.release()
            cv2.destroyAllWindows()

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
