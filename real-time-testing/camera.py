import cv2
import numpy as np
import copy 
from keras.models import load_model
from tools import remove_background, predict_rgb_image_vgg
model = load_model("../models/VGG_cross_validated.h5") # open saved model/weights from .h5 file

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

isBgCaptured = False  # bool, whether the background captured
bgModel = None

# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
	ret, frame = camera.read()
	frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
	frame = cv2.flip(frame, 1)  # flip the frame horizontally
	cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
				  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

	cv2.imshow('original', frame)

	if isBgCaptured == True:
		img = remove_background(frame, bgModel, learningRate)
		img = img[0:int(cap_region_y_end * frame.shape[0]),
			  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

		# convert the image into binary image
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

		ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		cv2.imshow('threshold', thresh)

		# get the contours
		thresh1 = copy.deepcopy(thresh)
		contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		length = len(contours)
		maxArea = -1
		if length > 0:
			for i in range(length):  # find the biggest contour (according to area)
				temp = contours[i]
				area = cv2.contourArea(temp)
				if area > maxArea:
					maxArea = area
					ci = i

			res = contours[ci]
			hull = cv2.convexHull(res)
			drawing = np.zeros(img.shape, np.uint8)
			cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
			cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

		cv2.imshow('contours', drawing)

	# Keyboard OP
	k = cv2.waitKey(10)

	if k == 27:  # press ESC to exit all windows at any time
		break
	elif k == ord('b'):  # press 'b' to capture the background
		bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
		isBgCaptured = True
		print('Background captured')
	elif k == ord('r'):  # press 'r' to reset the background
		time.sleep(1)
		bgModel = None
		isBgCaptured = False
		print('Reset background')
	elif k == 32:
		# If space bar pressed
		cv2.imshow('original', frame)
		# copies 1 channel BW image to all 3 RGB channels
		target = np.stack((thresh,) * 3, axis=-1)
		target = cv2.resize(target, (224, 224))
		target = target.reshape(1, 224, 224, 3)
		prediction, score = predict_rgb_image_vgg(target, model)
