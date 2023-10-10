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

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        # Draw the landmarks
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        # Calculate the center of the face
        x_center = face.left() + (face.width() // 2)
        y_center = face.top() + (face.height() // 2)

        # Calculate tilt angle based on the eyes' positions
        left_eye = face_landmarks.part(36)
        right_eye = face_landmarks.part(45)
        angle = compute_angle(left_eye, right_eye)

        # Rotate the vertical line by the calculated angle
        M = cv2.getRotationMatrix2D((x_center, y_center), -angle, 1)
        x1, y1 = M @ [x_center, face.top(), 1]
        x2, y2 = M @ [x_center, face.bottom(), 1]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        cv2.line(frame, (face.left(), y_center), (face.right(), y_center), (0, 255, 0), 1)

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
