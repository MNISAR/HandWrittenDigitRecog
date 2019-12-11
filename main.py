# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:51:43 2019

@author: monil
"""
import tensorflow as tf
graph = tf.get_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from tensorflow.contrib.layers import dropout as Dropout
from keras.optimizers import RMSprop
import keras
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL

class MODEL:
    def make_model(self):
        #keras.backend.clear_session()
        modelo = Sequential()
        modelo.add(Conv2D(64,  (3,3), padding='same', input_shape=(28, 28, 1)))
        modelo.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
        modelo.add(LeakyReLU(alpha=0.1))
        modelo.add(MaxPooling2D(2, 2))
        modelo.add(Dropout(0.2))
        modelo.add(Conv2D(128, (3,3), padding='same'))
        modelo.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
        modelo.add(LeakyReLU(alpha=0.1))
        modelo.add(Conv2D(128, (3,3), padding='same'))
        modelo.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
        modelo.add(LeakyReLU(alpha=0.1))
        modelo.add(MaxPooling2D(2,2))
        modelo.add(Dropout(0.2))
        modelo.add(Conv2D(256, (3,3), padding='same'))
        modelo.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
        modelo.add(LeakyReLU(alpha=0.1))
        modelo.add(Conv2D(256, (3,3), padding='same'))
        modelo.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
        modelo.add(LeakyReLU(alpha=0.1))
        modelo.add(MaxPooling2D(2,2))
        modelo.add(Dropout(0.2))
        modelo.add(Flatten())
        modelo.add(Dense(256))
        modelo.add(LeakyReLU(alpha=0.1))
        modelo.add(BatchNormalization())
        modelo.add(Dense(10, activation='softmax'))
        return modelo

    def preprocess_image(self,image):
        pass

    def loadmodel(self,model_path="models/model.h5"):
        self.model = self.make_model()
        self.model.load_weights(model_path)
        initial_learningrate=2e-3
        self.model.compile(loss= 'categorical_crossentropy', optimizer= keras.optimizers.RMSprop() , metrics=['accuracy'])

    def image_show(self,image):
        from PIL import Image
        image.reshape((28,28,1))
        image = image.astype('int8')
        img = Image.fromarray(image)
        img.show()


    ### load data and train model
    def predict(self,image): # assuming the image size is 28 X 28
        test = image.reshape(1, 28, 28, 1)
        test = test.astype('float32')
        test = test / 255.0
        print(test.shape)
        with graph.as_default():
            prediction = np.argmax(self.model.predict(test))
        print("Prediction: ",prediction)
        return prediction