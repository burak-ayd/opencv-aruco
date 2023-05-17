#!/usr/bin/env python3
import os
import numpy as np
import liboCams
import cv2
import time
import sys

# camera path ayarlanmalı
devpath = '/dev/v4l/by-id/usb-WITHROBOT_Inc._oCamS-1MGN-U_SN_2E955004-video-index0'

#liboCams.FindCamera('oCam')


if devpath is None:
  exit()

test = liboCams.oCams(devpath, verbose=1)

fmtlist = test.GetFormatList()

ctrlist = test.GetControlList()

test.Close() 

test = liboCams.oCams(devpath, verbose=0)
test.Set(fmtlist[7])
name = test.GetName()
test.Start()

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

try:
  while True:
    frameLeft,frameRight = test.GetFrame(mode=2)
    
    frameRightColor=cv2.cvtColor(frameRight, cv2.COLOR_BayerGR2RGB)
    frameLeftColor=cv2.cvtColor(frameLeft, cv2.COLOR_BayerGR2RGB)
    
    frameLeftGray= cv2.cvtColor(frameLeftColor, cv2.COLOR_BGR2GRAY)
    frameRightGray = cv2.cvtColor(frameRightColor, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(20, (8, 8)).apply(frameRightGray)
    blur = cv2.GaussianBlur(clahe, (7,7), 0)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,4)
    
    resizeLeft=cv2.resize(frameLeftGray,(840,840))
    resizeRight=cv2.resize(frameRightGray,(840,840))
    
    # Aruco ayarları
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_50"])
    
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids_r, rejected) = cv2.aruco.detectMarkers(adaptive_threshold, arucoDict,
      parameters=arucoParams)
    
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids_l, rejected) = cv2.aruco.detectMarkers(frameLeftGray, arucoDict,
      parameters=arucoParams)
    
    cv2.putText(resizeRight, f'ID: {ids_r}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255, 255, 255), 1)
    cv2.putText(resizeLeft, f'ID: {ids_l}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255, 255, 255), 1)
    
    mergedLeftRight = np.concatenate((resizeRight,resizeLeft), axis=1)
    cv2.imshow("Left - Right", mergedLeftRight)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
except Exception as error:
    # handle the exception
    print("An exception occurred:", error) 

test.Stop()  
cv2.destroyAllWindows()
char = cv2.waitKey(1)
test.Close()
