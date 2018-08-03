# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import subprocess

from glob import glob
import os

images = glob('extracted_images/*.bmp')




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=False,
                help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

name = 0
for img in images:
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    tlx, tly, brx, bry = 0, 0, 0, 0
    cx, cy = 1, 1
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        counter = 0

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        tempy = 0
        for (x, y) in shape:

            counter += 1
            if (counter == 3):
                # tempy = y
                tlx, tly = x, y
            # if (counter == 4):
            #     tlx, tly = x, int(y - (y - tempy)*0.15)
            if counter == 13:
                brx = x
            if counter == 11:
                bry = y
            # cv2.putText(image, "{}".format(counter), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
            # cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            if counter == 31:
                cx, cy = x, y

        counter = 0

    # cv2.rectangle(image, (tlx, tly), (brx, bry), (0, 0, 0), thickness=-1)
    if rects != None:
        crop_img = image[max(0, (cy - 128)):max(256, (cy + 128)), max(0, (cx -128)):max(256, (cx + 128))]
    # show the output image with the face detections + facial landmarks
    # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    # cv2.imshow("Output", crop_img)
    if not(os.path.exists('crop_img')):
        # Create directory
        subprocess.call('mkdir -p ' + 'crop_img', shell=True)
    cv2.imwrite('crop_img/{:05}.jpg'.format(name), crop_img)

    name+=1
    # os.remove(img)
# cv2.waitKey(0)
