# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 00:37:32 2019

@author: monil
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd

trainFile = "./data/train.csv"
testFile = "./data/test.csv"

df = pd.read_csv(trainFile, header =0)
trainData = df[:50000].drop('label', axis=1).to_numpy()
trainLabel = df[:50000][['label']].to_numpy()
testData = df[50000:].drop('label', axis=1).to_numpy()
testLabel = df[50000:][['label']].to_numpy()

image_length = len(trainData[0])
num_channels = 1
num_classes = 10

def build_model():
    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=()))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

