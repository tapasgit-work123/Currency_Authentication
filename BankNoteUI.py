# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:29:21 2022

@author: Tapas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:20:42 2022

@author: Tapas
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All'

@app.route('/predict',methods=['Get'])
def predict_note_authentication():
    
    """ Let's Authenticate Bank note
    This is using docstrings for specifications
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
              
    """
      
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is "+ str(prediction)


if __name__=='__main__':
    app.run()