#!/usr/bin/env python3
import math
import os
import sys
import time

import cv2
import numpy as np

import liboCams

# camera path ayarlanmalı
devpath = '/dev/v4l/by-id/usb-WITHROBOT_Inc._oCamS-1MGN-U_SN_2E955004-video-index0'

# liboCams.FindCamera('oCam')


if devpath is None:
    exit()

test = liboCams.oCams(devpath, verbose=1)

fmtlist = test.GetFormatList()

ctrlist = test.GetControlList()

test.Close()

test = liboCams.oCams(devpath, verbose=0)
test.Set(fmtlist[0])
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
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

try:
    while True:
        frameLeft, frameRight = test.GetFrame(mode=2)

        frameRightColor = cv2.cvtColor(frameRight, cv2.COLOR_BayerGR2RGB)
        frameLeftColor = cv2.cvtColor(frameLeft, cv2.COLOR_BayerGR2RGB)

        frameLeftGray = cv2.cvtColor(frameLeftColor, cv2.COLOR_BGR2GRAY)
        frameRightGray = cv2.cvtColor(frameRightColor, cv2.COLOR_BGR2GRAY)

        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

        cornerRefinementMethod = [
            cv2.aruco.CORNER_REFINE_NONE,
            cv2.aruco.CORNER_REFINE_SUBPIX,
            cv2.aruco.CORNER_REFINE_CONTOUR,
        ]

        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoParams.minDistanceToBorder = 0
        arucoParams.cornerRefinementMethod = cornerRefinementMethod[1]
        arucoParams.cornerRefinementWinSize = 5
        arucoParams.cornerRefinementMaxIterations = 30

        (corners, ids, rejected) = cv2.aruco.detectMarkers(frameLeftColor, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            # otomatik olarak çizgi çizdiriyor
            cv2.aruco.drawDetectedMarkers(frameLeftColor, corners, ids)

            # manuel olarak çizgi çizdiriyor
            # ids = ids.flatten()
            # aruco_count = np.size(ids)
            # # loop over the detected ArUCo corners
            # for markerCorner, markerID in zip(corners, ids):
            #     # extract the marker corners (which are always returned in
            #     # top-left, top-right, bottom-right, and bottom-left order)
            #     corners = markerCorner.reshape((4, 2))
            #     (topLeft, topRight, bottomRight, bottomLeft) = corners
            #     # convert each of the (x, y)-coordinate pairs to integers
            #     topRight = (int(topRight[0]), int(topRight[1]))
            #     bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            #     bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            #     topLeft = (int(topLeft[0]), int(topLeft[1]))
            #     # draw the bounding box of the ArUCo detection
            #     cv2.line(frameLeftColor, topLeft, topRight, (0, 255, 0), 2)
            #     cv2.line(frameLeftColor, topRight, bottomRight, (0, 255, 0), 2)
            #     cv2.line(frameLeftColor, bottomRight, bottomLeft, (0, 255, 0), 2)
            #     cv2.line(frameLeftColor, bottomLeft, topLeft, (0, 255, 0), 2)
            #     # compute and draw the center (x, y)-coordinates of the ArUco
            #     # marker
            #     cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            #     cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            #     cv2.circle(frameLeftColor, (topLeft[0], topLeft[1]), 4, (0, 0, 255), -1)
            #     # draw the ArUco marker ID on the image
            #     cv2.putText(
            #         frameLeftColor,
            #         str(markerID),
            #         (topLeft[0], topLeft[1] - 15),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 255, 0),
            #         2,
            #     )
            print("[INFO] ArUco marker ID: {}".format(ids))
        cv2.imshow("Image", frameLeftColor)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as error:
    # handle the exception
    print("An exception occurred:", error)

test.Stop()
cv2.destroyAllWindows()
char = cv2.waitKey(1)
test.Close()
