"""import pandas as pd

df = pd.read_csv('data/train.csv', header=0)
df = df.iloc[0]
label = df['label']
test = df.drop(['label']).to_numpy()
test = test.reshape( 28, 28, 1)
test = test.astype('float32')
test = test / 255.0
print(label)
print(test)
"""
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from PIL import ImageFilter
import PIL
import numpy as np

im = PIL.Image.open('static/image2.png')
im = im.filter(PIL.ImageFilter.ModeFilter(size=3)).resize((28,28))

#im_arr = im_arr[:,:,0].reshape((im_arr.shape[0], im_arr.shape[1], 1))

#im_arr[:,:,1] = np.zeros(im_arr.shape[:2])
#im_arr[:,:,2] = np.zeros(im_arr.shape[:2])
#im_arr[:,:,0] = np.zeros(im_arr.shape[:2])


#im_arr = np.sum(im_arr, axis=2)

#print(im_arr.shape)
#im = PIL.Image.fromarray(im_arr)
im.show()




"""df = pd.read_csv('data/train.csv', header=0)
df = df.iloc[0]
label = df['label']
test = df.drop(['label']).to_numpy()
test = test.reshape((28, 28))
test = test.astype('int8')

img = Image.fromarray(test)
imagebox = img.getbbox()
img = img.crop(imagebox)
img.show()"""



