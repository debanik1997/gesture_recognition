import os
import warnings
import cv2
import tensorflow.keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# change this with naming schematic
gestures = {'wave':'Hello',
			'pointup':'Volume_Up',
			'rockon':'Play',
			'pointright':'Next'}

gestures_map = {'Hello' : 0,
				'Volume_Up': 1,
				'Play': 2,
				'Next': 3,
				}

def process_image(path):
	img = Image.open(path)
	img = img.resize((224, 224))
	img = np.array(img)
	return img

def process_data(X_data, y_data):
	X_data = np.array(X_data, dtype = 'float32')
	if rgb:
		pass
	else:
		X_data = np.stack((X_data,)*3, axis=-1)
	X_data /= 255
	y_data = np.array(y_data)
	y_data = to_categorical(y_data, num_classes=5)
	return X_data, y_data

def walk_file_tree(relative_path):
	X_data = []
	y_data = [] 
	for directory, subdirectories, files in os.walk(relative_path):
		for file in files:
			if not file.startswith('.') and (not file.startswith('C_')):
				path = os.path.join(directory, file)
				gesture_name = gestures[file.split('_')[0]]
				y_data.append(gestures_map[gesture_name])
				X_data.append(process_image(path))   

			else:
				continue

	X_data, y_data = process_data(X_data, y_data)
	return X_data, y_data

class Data(object):
	def __init__(self):
		self.X_data = []
		self.y_data = []

	def get_data(self):
		return self.X_data, self.y_data

if __name__ == "__main__":
	print("START")
	relative_path = './data_creation/data/'
	rgb = True
	X_data, y_data = walk_file_tree(relative_path) # X_data = labelling; y_data = images
	silhouette = Data()
	silhouette.X_data, silhouette.y_data = walk_file_tree(relative_path)

	# X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb = train_test_split(image_rgb, y_data, test_size = 0.2, random_state=12, stratify=y_data)
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=12, stratify=y_data)
	file_path = './models/temp_model.h5'
	model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True)

	early_stopping = EarlyStopping(monitor='val_accuracy',
								min_delta=0,
								patience=10,
								verbose=1,
								mode='auto')
	#,restore_best_weights=True
	imageSize = 224

	vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
	optimizer1 = optimizers.Adam()

	base_model = vgg_base  # Topless
	
	# Adding top layers
	x = base_model.output
	x = Flatten()(x)
	x = Dense(128, activation='relu', name='fc1')(x)
	x = Dense(128, activation='relu', name='fc2')(x)
	x = Dense(128, activation='relu', name='fc3')(x)
	x = Dropout(0.5)(x) # turns off activation of neurons during training
	x = Dense(64, activation='relu', name='fc4')(x)
	predictions = Dense(5, activation='softmax')(x)

	# new model with top layers and predictions
	model = Model(inputs=base_model.input, outputs=predictions)

	# Train top layers only
	for layer in base_model.layers:
	    layer.trainable = False

	callbacks_list = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)]

	# loss = loss value to be minimized; metrics = list of metrics to be evaluated during training and testing 
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# (input data, target data, )
	model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping, model_checkpoint])
