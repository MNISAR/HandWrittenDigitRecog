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


app = Flask(__name__,static_url_path='/static')   # Flask constructor 
m = MODEL()
m.make_model()
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
    print(string)
    
    image = base64.b64decode(string)
    im  = Image.open(BytesIO(image)).convert('1')
    im = im.resize((28,28))
    im = np.array(im)
    raveled = []
    for i in im:
        raveled.extend(i)
    raveled = np.array(raveled).reshape((1,len(raveled)))
    print(len(raveled))
    print("Prediction:",m.predict(raveled))
    return ""
 
if __name__=='__main__': 
   app.run(debug=True) 
   