#!/usr/bin/env python3
import os
import numpy as np
import liboCams
import cv2
import time
import sys

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-a", "--alltest", dest="alltest",
                  action='store_true', help="test all resolution in playtime")
parser.add_option("-t", "--time", dest="playtime", default = 1, type = "int",
                  help="playtime for streaming [sec] intValue, 0 means forever")
parser.add_option("-i", "--index", dest="index", default = 0, type = "int",
                  help="index of resolusion mode")
(options, args) = parser.parse_args()

# camera path ayarlanmalı
devpath = '/dev/v4l/by-id/usb-WITHROBOT_Inc._oCamS-1MGN-U_SN_2E955004-video-index0'

#liboCams.FindCamera('oCam')


if devpath is None:
  exit()

test = liboCams.oCams(devpath, verbose=1)

fmtlist = test.GetFormatList()

ctrlist = test.GetControlList()

test.Close()
if options.alltest is True:
  len_range = list(range(len(fmtlist)))
else:
  if options.index >= len(fmtlist):
    print('INDEX error', options.index, 'index reset to default value 0')
    options.index = 0  
  len_range = { options.index }  

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
    (corners, ids_r, rejected) = cv2.aruco.detectMarkers(frameRightColor, arucoDict,
      parameters=arucoParams)
    
    cv2.putText(resizeRight, f'ID: {ids_r}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 255), 1)
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
