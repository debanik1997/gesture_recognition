import cv2
import numpy as np
from gtts import gTTS 
  
# This module is imported so that we can  
# play the converted audio 
import os 

gesture_names = {0: 'Hello Alexa',
				 1: 'Volume Up',
				 2: 'Play music',
				 3: 'Next song'}

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

	# model.predict() returns an array of probabilities - 
	# np.argmax grabs the index of the highest probability.
	result = gesture_names[np.argmax(pred_array)]
	
	# A bit of magic here - the score is a float, but I wanted to
	# display just 2 digits beyond the decimal point.
	score = float("%0.2f" % (max(pred_array[0]) * 100))
	print('Result: ' + str(result) + ', Score: ' + str(score))
	# The text that you want to convert to audio 
	mytext = result
	  
	# Language in which you want to convert 
	language = 'en'
	  
	# Passing the text and language to the engine,  
	# here we have marked slow=False. Which tells  
	# the module that the converted audio should  
	# have a high speed 
	myobj = gTTS(text=mytext, lang=language, slow=False) 
	  
	# Saving the converted audio in a mp3 file named 
	# welcome  
	myobj.save("welcome.mp3") 
	  
	# Playing the converted file 
	os.system("mpg321 welcome.mp3") 
	return result, score
