# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:51:43 2019

@author: monil
"""
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MODEL:
    def make_model(self):
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
        self.model.compile(loss= 'categorical_crossentropy', optimizer= RMSprop(lr=initial_learningrate) , metrics=['accuracy'])

    def image_show(self,image):
        from PIL import Image
        image.reshape((28,28))
        image = image.astype('int8')
        img = Image.fromarray(image)
        img.show()


    ### load data and train model
    def predict(self,image):
        #image = self.preprocess_image(image)  ## gives a 784 X 1 array which we will reshape

        test = image.reshape(1, 28, 28, 1)
        test = test.astype('float32')
        test = test / 255.0
        self.image_show(image)
        prediction = np.argmax(self.model.predict(test))
        print("Prediction: ",prediction)
        return prediction