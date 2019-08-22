import cv2
import os
import sys
from string import Template

# first argument is the haarcascades path
cap = cv2.VideoCapture('F:/Mycompleted task/Face_Recognition/Mutil_person/videoplayback.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

scale_factor = 1.1
min_neighbors = 3
min_size = (30, 30)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE
while(cap.isOpened()):
    ret, frame = cap.read()
    image =frame

    faces = face_cascade.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                          minSize=min_size, flags=flags)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.imshow('outimage', image)
        cv2.waitKey(0)
