import cv2
import numpy as np

def remove_background(frame, background_model, learningRate):
	fgmask = background_model.apply(frame, learningRate=learningRate)
	kernel = np.ones((3, 3), np.uint8)
	fgmask = cv2.erode(fgmask, kernel, iterations=1)
	res = cv2.bitwise_and(frame, frame, mask=fgmask)
	return res

def predict_rgb_image_vgg(image, model):
	image = np.array(image, dtype='float32')
	image /= 255
	pred_array = model.predict(image)

	gesture_names = {0: 'C',
				 1: 'Fist',
				 2: 'L',
				 3: 'Okay',
				 4: 'Palm',
				 5: 'Peace'}

	# model.predict() returns an array of probabilities - 
	# np.argmax grabs the index of the highest probability.
	result = gesture_names[np.argmax(pred_array)]
	
	# A bit of magic here - the score is a float, but I wanted to
	# display just 2 digits beyond the decimal point.
	score = float("%0.2f" % (max(pred_array[0]) * 100))
	print('Result: ' + str(result) + ', Score: ' + str(score))
	return result, score
