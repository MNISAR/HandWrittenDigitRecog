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
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

## viewing the image from dataset
#for i in image:
#	for j in i:
#		print(j, end=' ')
#	print("")
#plt.imshow(image,cmap='Greys')
#plt.show()	
string = 'iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAA/0lEQVRIS+2WsRLCMAxD1X+CHWY+CvgoZmCmn8MOJy7hTJs4SgoduHbk1DxbdkU6zPx0M/OwAOn4QbRd0imWPkTgEcAOwNrTe8AbgJUIszIX7AFp0R7ABcC1AGZnp6CP0uTZOWCE8eW+ZJMphq7cAWzCb+z2Y7YKcPRSoVtbrAS0s6uFxa3mKPhIwHOwhLPbNixNVYets7N1yUBXWNHpAnyb9f+WFr+hbyzOMGl+bqsHrMnQYfPZwlNZGsUtsRbB2XhMAadmqd0FKUunztEWPBqLZ2ky7YVNjeGfHIn3j2/vMsrdZ/hZJZdOuWLwIBVILQvN6msOEtx8SbgD2SvjEwleVB0a8ZnJAAAAAElFTkSuQmCC'
image = base64.b64decode(string)
#print(image)
i = BytesIO(image)
image = mpimg.imread(i, format='jpg')
Image.fromarray(np.array(image)).save("static/result.png", format="png")
im = Image.fromarray(np.array(image)[:,:,3])
im.thumbnail((14,14))
image = np.array(im)

## reading image int numpy array
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

bwimg = np.array(image)
"""imgtonp = np.array(plt.imread("static/cbimage.jpg"))"""
"""if(imgtonp.shape!=(56,56)):
	print("Not 28 X 28")
	from PIL import Image
	#img = Image.fromarray(imgtonp)
	img = Image.open("static/cbimage.jpg")
	img.show()
	#rsize = img.resize((28,28))
	rsizeArr = rgb2gray(np.asarray(rsize))
	print(rsizeArr.shape)
	bwimg = rsizeArr

else:

	imgplot = plt.imshow(imgtonp, interpolation="bicubic")
	plt.show()
	bwimg = []

	for i in range(imgtonp.shape[0]):
		temp = []
		for j in range(imgtonp.shape[1]):
			temp.append(np.sum(imgtonp[i,j,:]))
		bwimg.append(temp)

	#plt.imshow(bwimg, cmap='Greys')
"""
for k in range(1):
	for i in bwimg:
		for j in i:
			print("%3d"%j, end=' ')
		print("")
	print("-"*10)
"""
img = mpimg.imread('static/cbimage.jpg')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#plt.show()
"""



#im = PIL.Image.open('static/cbimage.png')
#rgb_im = im.convert('RGB')
#rgb_im.save('static/cbimage.jpg')

#im = im.filter(PIL.ImageFilter.MinFilter(size=3))

#im_arr = im_arr[:,:,0].reshape((im_arr.shape[0], im_arr.shape[1], 1))

#im_arr[:,:,1] = np.zeros(im_arr.shape[:2])
#im_arr[:,:,2] = np.zeros(im_arr.shape[:2])
#im_arr[:,:,0] = np.zeros(im_arr.shape[:2])


#im_arr = np.sum(im_arr, axis=2)

#print(im_arr.shape)
#im = PIL.Image.fromarray(im_arr)





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



