import cv2
import dlib
import math

def compute_angle(point1, point2):
    return math.degrees(math.atan2(point2.y - point1.y, point2.x - point1.x))

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

curr_frame = None
M = 0.0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        left_eye = face_landmarks.part(36)
        right_eye = face_landmarks.part(45)
        angle = compute_angle(left_eye, right_eye)

        x_center = face.left() + (face.width() // 2)
        y_center = face.top() + (face.height() // 2)


        # Rotate the entire image
        M = cv2.getRotationMatrix2D((x_center, y_center), -angle, 1)
        # frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        curr_frame = frame

    cv2.imshow("Adjusted Image", frame)

    key = cv2.waitKey(1)
    if key == ord('t') and curr_frame is not None:
        # Constants for US passport photo
        PASSPORT_WIDTH = 600  # in pixels
        PASSPORT_HEIGHT = 600  # in pixels
        FACE_HEIGHT_RATIO = 0.75  # desired face height / photo height
        EYES_POSITION_RATIO = 1/3  # position from the bottom

        # Calculate scale factor to get desired face height
        face_height = face.bottom() - face.top()
        desired_face_height = int(FACE_HEIGHT_RATIO * PASSPORT_HEIGHT)
        scale_factor = desired_face_height / face_height

        # Resize the entire image
        t_frame = cv2.resize(curr_frame, (int(curr_frame.shape[1] * scale_factor), int(curr_frame.shape[0] * scale_factor)))

        # Calculate the position to crop
        x_center = int(face.left() * scale_factor + (face.width() * scale_factor) / 2)
        y_center = int(face.top() * scale_factor + (face.height() * scale_factor) / 2)

        top_left_y = int(y_center - (PASSPORT_HEIGHT * EYES_POSITION_RATIO))
        bottom_right_y = top_left_y + PASSPORT_HEIGHT
        top_left_x = x_center - (PASSPORT_WIDTH // 2)
        bottom_right_x = top_left_x + PASSPORT_WIDTH

        # Crop the image
        cropped_passport = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cropped_passport = cv2.warpAffine(frame, M, (cropped_passport.shape[1], cropped_passport.shape[0]))

        cv2.imshow("Passport Photo", cropped_passport)
        cv2.waitKey(0)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
