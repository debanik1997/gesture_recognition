from sklearn.metrics import classification_report, confusion_matrix
import tensorflow
import keras
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

def save_augmented_images(generator, img_path, test_batchsize, directory, prefix):
    
    x = load_img(img_path)
    x = img_to_array(x)
    x = x.reshape((1, ) + x.shape)
    
    i = 0
    for batch in generator.flow(x, batch_size=test_batchsize,
                          save_to_dir=directory, save_prefix=prefix, save_format='jpeg'):
        i += 1
        if i > 20:
            break

if __name__ == "__main__":

    image_size = 224
    test_dir = './data_creation/data'
    test_batchsize = 10
    
    # generate batches of tensor image data with real-time data augmentation
    test_datagen = ImageDataGenerator(rescale=1./255, 
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2) 

    # saves augmented images to a directory
    
    # save_augmented_images(test_datagen, './data_creation/data/Hello/wave_0.png', test_batchsize, './data_creation/data_augmented/Hello_augmented', 'wave')
    
    # generates batches of augmented data, precisely, 6 batches of size 10
    test_generator = test_datagen.flow_from_directory( 
        test_dir,
        target_size=(image_size, image_size),
        batch_size=test_batchsize,
        class_mode='categorical',
        shuffle=False)

    vgg_model = tensorflow.keras.models.load_model('./models/trained_model.h5')

    ### Confusion Matrix and Classification Report

    # generates predictions for the input samples from a data generator 
    Y_pred = vgg_model.predict_generator( 
        test_generator, test_generator.samples // test_generator.batch_size)
    y_pred = np.argmax(Y_pred, axis=1)

    print(test_generator.classes)
    print(y_pred)
    
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    
    print('Classification Report')
    target_names = ['Hello', 'Next', 'Play', 'Volume_Up']
    print(classification_report(test_generator.classes,
                                y_pred, target_names=target_names))

    # Evaluating using Keras model_evaluate:
    x, y = zip(*(test_generator[i] for i in range(len(test_generator))))
    x_test, y_test = np.vstack(x), np.vstack(y)
    
    print(x_test.shape)
    print(y_test.shape)

    loss, acc = vgg_model.evaluate(x_test, y_test, batch_size=64)

    print("Accuracy: ", acc)
    print("Loss: ", loss)
