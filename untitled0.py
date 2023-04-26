#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:55:03 2023

@author: sentieo
"""
import numpy as np
import pickle
import streamlit as st
from PIL import Image

#image = Image.open('/home/sentieo/Deploying_ml/image.jpg')
st.image("https://s3-us-west-2.amazonaws.com/uw-s3-cdn/wp-content/uploads/sites/6/2017/11/04133712/waterfall.jpg"width=400, # Manually Adjust the width of the image as per requirement)
#st.image(image)

loaded_model = pickle.load(open('/home/sentieo/Deploying_ml/trained_model(1).sav', 'rb'))




# creating a function for Prediction

def heartattack_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not prone to heart attack'
    else:
      return 'The person is prone to heart attack'
  
def main():
    
    
    # giving a title
    st.title('Heart Attack Analysis Web App')
    
    
    # getting the input data from the user
    
    
    age = st.text_input('Age of the person')
    sex = st.text_input('Sex(Value 0: Female, Value 1: Male)')
    RestingBP = st.text_input('Blood Pressure value')
    chol = st.text_input('Cholestrol Level')
    MaxHeartRate = st.text_input('Maximum Heart Rate')
    exng = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    ChestPain = st.text_input('Chest Pain level(Value 1: typical angina Value 2: atypical angina Value 3: non-anginal pain Value 4: asymptomatic)')
    fbs = st.text_input('Fasting Blood Sugar value(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Analysis Result'):
        diagnosis = heartattack_prediction([age, sex , RestingBP, chol, MaxHeartRate,exng , ChestPain, fbs])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
