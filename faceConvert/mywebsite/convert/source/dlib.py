import cv2
import dlib
import numpy as np

def get_people_num(file_name):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./convert/source/models/shape_predictor_68_face_landmarks.dat')

    ff = np.fromfile(file_name, np.uint8)
    img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    return len(faces)