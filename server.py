# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:25:38 2019

@author: monil
"""

from flask import Flask,render_template,request,jsonify
import base64
from PIL import Image
from io import BytesIO
from main import MODEL
import numpy as np

app = Flask(__name__,static_url_path='/static')   # Flask constructor 
m = MODEL()
m.loadmodel()
# A decorator used to tells the application 
# which URL is associated function 
@app.route('/')      
def hello():
    doc = ''
    with open('templates/mainPage.html') as f:
    	doc = f.read()
    return doc
  
    
@app.route('/process_request', methods=['POST', 'GET'])
def prediction():
    imagefile = request.form['imagefile']
    string = "".join(imagefile.split(",")[1:])
    image = base64.b64decode(string)
    im  = Image.open(BytesIO(image)).convert('L')
    im.thumbnail((28,28))
    image_numpy = np.array(im)
    print(image_numpy.shape)
    image_numpy.reshape((28,28,1))
    image_numpy = image_numpy.astype('int8')
    print(m.predict(image_numpy))
    return ""
 
if __name__=='__main__': 
   app.run(debug=True) 
   