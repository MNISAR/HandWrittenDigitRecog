# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:25:38 2019

@author: monil
"""
from flask import Flask,render_template,request,jsonify

import numpy as np


app = Flask(__name__,static_url_path='/static')   # Flask constructor 

# A decorator used to tells the application 
# which URL is associated function 
@app.route('/')      
def hello():
    doc = ''
    with open('templates/mainPage.html') as f:
    	doc = f.read()
    return doc
 
if __name__=='__main__': 
   app.run(debug=True, host='0.0.0.0')
   