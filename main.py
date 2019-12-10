# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:51:43 2019

@author: monil
"""
import numpy as np
import pandas as pd
from PIL import Image
import PIL

class MODEL:
    def make_model(self):
        df = pd.read_csv("data/train.csv")
        y_train = df['label'].to_numpy()
        X_train = df.drop(['label'], axis=1).to_numpy()

        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        self.model = model

    def preprocess_image(self,image):
        pass

    def image_show(self,image):
        image.reshape((28,28,1))
        image = image.astype('int8')
        img = PIL.Image.fromarray(image)
        img.show()


    ### load data and train model
    def predict(self,image):
        prediction = self.model.predict(image)
        return prediction

