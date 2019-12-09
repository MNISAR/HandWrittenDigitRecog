# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:25:38 2019

@author: monil
"""

from flask import Flask,render_template
app = Flask(__name__)   # Flask constructor 

# A decorator used to tells the application 
# which URL is associated function 
@app.route('/')      
def hello(): 
    #return '<html><head><title>TITLE</title></head></html>'
    doc = ''
    with open('webpages/mainPage.html') as f:
    	doc = f.read()
    return doc
  
    
@app.route('/process_request')
def prediction():
    return "check CMD"
 
 
if __name__=='__main__': 
   app.run(debug=True) 
   