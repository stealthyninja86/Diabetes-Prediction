# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/naray/Desktop/diabetesPrediction/trained_model.sav','rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)
arr = np.asarray(input_data)
input_data_reshaped = arr.reshape(1,-1)
#std_data = scaler.transform(input_data_reshaped)
#print(std_data)

pred = loaded_model.predict(input_data_reshaped)
print(pred)

if(pred[0] == 0):
  print("not diabetic")
else:
  print("diabetic")