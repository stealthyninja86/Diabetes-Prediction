# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:16:30 2022

@author: naveen
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav','rb'))


def diabetes_prediction(input_data):
    arr = np.asarray(input_data)
    input_data_reshaped = arr.reshape(1,-1)
    
    pred = loaded_model.predict(input_data_reshaped)
    print(pred)
    
    if(pred[0] == 0):
     return "not diabetic"
    else:
      return "diabetic"
  
    
def main():
    
    #title
    st.title('Diabetes Prediction')
    
    #getting input data from user
    
    Pregnancies = st.text_input('number of pregnancies')
    Glucose = st.text_input('glucose level')
    BloodPressure = st.text_input('blood pressure value')
    SkinThickness = st.text_input('skin thickness')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
