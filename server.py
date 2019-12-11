# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:25:38 2019

@author: monil
"""

from flask import Flask,render_template,request,jsonify
import base64
from PIL import Image
import PIL
from io import BytesIO
from main import MODEL
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
    image = mpimg.imread(BytesIO(image), format='jpg')
    Image.fromarray(np.array(image)).save("static/result.png", format="png")
    image = np.array(image)[:,:,3]
    im = Image.fromarray(image)
    im.thumbnail((28,28))
    image = np.array(im)
    pr = m.predict(image)
    return render_template('result.html', prediction=str(pr), img="static/result.png?"+str(datetime.now()))
