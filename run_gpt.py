import cv2
import dlib
import numpy as np

# Load the predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read the image
img = cv2.imread("face.jpg")
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    landmarks = predictor(image=gray, box=face)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    points = np.array(points)

    # Get the coordinates of the middle of the face horizontally
    top_mid = (points[21] + points[22]) // 2
    bottom_mid = points[8]

    # Get the coordinates of the middle of the face vertically
    left_mid = points[1]
    right_mid = points[15]

    # Draw a horizontal line
    cv2.line(img, tuple(top_mid), tuple(bottom_mid), (255, 0, 0), 2)

    # Draw a vertical line
    cv2.line(img, tuple(left_mid), tuple(right_mid), (255, 0, 0), 2)

# Display the image
cv2.imshow(winname="Face", mat=img)

# Wait for a key press and then close the displayed window
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
