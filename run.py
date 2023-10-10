import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        # Calculate the center of the face
        x_center = face.left() + (face.width() // 2)
        y_center = face.top() + (face.height() // 2)

        # Draw vertical and horizontal lines
        cv2.line(frame, (x_center, face.top()), (x_center, face.bottom()), (0, 255, 0), 1)
        cv2.line(frame, (face.left(), y_center), (face.right(), y_center), (0, 255, 0), 1)


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()