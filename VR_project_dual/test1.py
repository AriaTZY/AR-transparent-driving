import cv2 as cv
import numpy as np
import Feature.read as read
import pylab
from camero.camero_calibration import cali_live
import VR_project.config as cfg

cap2 = cv.VideoCapture(0)
cap3 = cv.VideoCapture(2)
while True:
    sucess2, img2 = cap2.read()
    sucess3, img3 = cap3.read()
    img2 = img2[60:420, :]  # opencv的读入格式长宽比有些问题，所以重新裁剪
    img3 = img3[60:420, :]
    # img2 = cali_live(img2)
    # img3 = cali_live(img3)
    cv.imshow('window2', img2)
    cv.imshow('window3', img3)
    a = cv.waitKey(1)
    if a == 115:  # save键
        cv.imwrite('left.jpg', img2)
        cv.imwrite('middle.jpg', img3)