from __future__ import print_function
import cv2
import argparse

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


cam = cv2.VideoCapture(1)
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

    # convert to grayscale
    if isColor == False:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type)
        blur = cv2.GaussianBlur(dst, (blur_value, blur_value), 0)
        if findContours:
            _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  # add this line
            output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
            print(str(len(contours))+"\n")
        else:
            output = blur

    else:
        _, dst = cv2.threshold(frame, threshold_value, max_binary_value, threshold_type)
        blur = cv2.GaussianBlur(dst, (blur_value, blur_value), 0)
        output = blur

    cv2.imshow(window_name, output)
    k = cv2.waitKey(1)  # k is the key pressed
    if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively
        # exit
        cv2.destroyAllWindows()
        cam.release()
        break
