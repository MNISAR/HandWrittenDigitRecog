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

df = pd.read_csv('data/train.csv', header=0)
df = df.iloc[0]
label = df['label']
test = df.drop(['label']).to_numpy()
test = test.reshape((28, 28))
test = test.astype('int8')
from PIL import Image
img = Image.fromarray(test)
imagebox = img.getbbox()
img = img.crop(imagebox)
img.show()