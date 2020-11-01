from __future__ import print_function
import cv2
import argparse
import numpy as np
import pyautogui
from scipy.spatial import ConvexHull

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Threshold Demo'
isColor = False


def nothing(x):
    pass


def isIncreased(value, prevValue, threshold=0):
    return (value > prevValue + threshold)


def isDecreased(value, prevValue, threshold=0):
    return (value < prevValue - threshold)


def ZoomIn():
    pyautogui.hotkey('command', '+')


def ZoomOut():
    pyautogui.hotkey('command', '-')


def RotateRight():
    pyautogui.hotkey('command', 'R')


def RotateLeft():
    pyautogui.hotkey('command', 'R', presses=3)


prevHullArea = 0
prevAngle = 0
counter = 0

cam = cv2.VideoCapture(0)
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_type, window_name, 3, max_type, nothing)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name, 0, max_value, nothing)
# Call the function to initialize
cv2.createTrackbar(trackbar_blur, window_name, 1, 20, nothing)
# create switch for ON/OFF functionality
color_switch = 'Color'
cv2.createTrackbar(color_switch, window_name, 0, 1, nothing)
cv2.createTrackbar('Contours', window_name, 0, 1, nothing)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
    blur_value = blur_value + (blur_value % 2 == 0)
    isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)

    # Part 1 : Extracting Hand from the feed [5 pts]
    # (Part 1) Separation of skin color with HSV and YCbCr
    lower_HSV = np.array([0, 40, 0], dtype="uint8")
    upper_HSV = np.array([25, 255, 255], dtype="uint8")

    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)

    lower_YCrCb = np.array((0, 138, 67), dtype="uint8")
    upper_YCrCb = np.array((255, 173, 133), dtype="uint8")

    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)

    skinMask = cv2.add(skinMaskHSV, skinMaskYCrCb)

    # (Part 1) Erosion and dialation transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    frame = skin  # (Part 1)

    # Part 2 : Processing the hand image with connected component analysis [5 pts]
    # (Part 2) threshold and binarize the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  # For Part 2

    # (Part 2) Connected components
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh, ltype=cv2.CV_16U)
    markers = np.array(markers, dtype=np.uint8)
    label_hue = np.uint8(179*markers/np.max(markers))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    # (Part 2) ROI
    statsSortedByArea = stats[np.argsort(stats[:, 4])]

    if (ret > 2):
        try:
            # (Part 2) ROI (continued)
            roi = statsSortedByArea[-3][0:4]
            x, y, w, h = roi
            subImg = labeled_img[y:y+h, x:x+w]
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY)

            # (Part 2) Find contours in the ring and fit an ellipse
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            maxCntLength = 0
            for i in range(0, len(contours)):
                cntLength = len(contours[i])
                if(cntLength > maxCntLength):
                    cnt = contours[i]
                    maxCntLength = cntLength
            if(maxCntLength >= 5):
                # (Part 2) Fit ellipse and get ellipse parameters
                ellipseParam = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipseParam
                # print('======================\n**Part 2**\nx: %s\ny: %s\nMA: %s\nma: %s\nangle: %s\n' % (x, y, MA, ma, angle))
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB)
                subImg = cv2.ellipse(subImg, ellipseParam, (0, 255, 0), 2)

            subImg = cv2.resize(subImg, (0, 0), fx=3, fy=3)
            # cv2.imshow("ROI "+str(2), subImg)
            # cv2.waitKey(1)
        except:
            # print("No hand found")
            pass

    # Part 3 : Tracking 2D finger positions
    # (Part 3) repeating some part 2 without inversion
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_OTSU)
    thresholdedHandImage = thresh
    # Part 3a: Processing the hand image with contour and hull analysis [5 pts]
    fingerCount = 0
    if (ret > 2):
        try:
            _, contours, _ = cv2.findContours(thresholdedHandImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            thresholdedHandImage = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            if len(contours) > 1:
                largestContour = contours[0]
                M = cv2.moments(largestContour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # print('**Part 3a**\ncX: %s\ncY: %s' % (cX, cY))
                hull = cv2.convexHull(largestContour, returnPoints=False)
                hullPoints = cv2.convexHull(largestContour, returnPoints=True)
                hullArea = cv2.contourArea(hullPoints)
                for cnt in contours[:1]:
                    defects = cv2.convexityDefects(cnt, hull)
                    if(not isinstance(defects, type(None))):
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            cv2.line(thresholdedHandImage, start, end, [0, 255, 0], 2)
                            cv2.circle(thresholdedHandImage, far, 5, [0, 0, 255], -1)
                    # Part 3b : Detecting fingers in the image [10 pts]
                    for i in range(len(defects)):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                        a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                        b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
                        angle = np.arccos((a_squared + b_squared - c_squared) / (2 * np.sqrt(a_squared * b_squared)))
                        if angle <= np.pi / 3:
                            if fingerCount:
                                fingerCount += 1
                            else:
                                fingerCount += 2
                            cv2.circle(thresholdedHandImage, far, 4, [0, 0, 255], -1)
            print("fingerCount: %s" % str(fingerCount))

        except:
            # print('no hand found')
            pass
    cv2.imshow(window_name, thresholdedHandImage)

    # (Part 4) Complex Gesture #1: Zoom in and zoom out based on hand’s convex hull size
    if isIncreased(hullArea, prevHullArea):
        ZoomIn()
    elif isDecreased(hullArea, prevHullArea):
        ZoomOut()
    prevHullArea = hullArea

    k = cv2.waitKey(1)  # k is the key pressed
    if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively
        # exit
        cv2.destroyAllWindows()][0]
        cam.release()
        break

    counter += 1
