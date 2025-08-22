import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler 


model = joblib.load("Sneak.pkl")
scaler = joblib.load("scalerX.pkl")
scalerY = joblib.load("scalerY.pkl")


st.title("Model Demo")
st.markdown("<h2 style = 'text-align: left; color: pink;'> Predicting Heating and Cooling Load to evaluate a building's Energy Efficiency</h2>", unsafe_allow_html = True)
st.markdown("<h4 style = 'text-align: left; color: gray;'> lowering a building's heating and cooling load is the key to using less energy)</h4>", unsafe_allow_html = True)
#st.subheader(":hotpink[", divider = "green")

with st.sidebar:
    X1 = st.number_input("Relative Compactnes", min_value = 0.62, max_value = 0.98)
    X2 = st.number_input("Surface Area", min_value = 514.5, max_value = 808.5)
    X3 = st.number_input("Wall Area", min_value = 245.0, max_value = 416.5)
    X4 = st.number_input("Roof Area", min_value = 110.25, max_value = 220.5)
    X5 = st.number_input("Overall Height", min_value = 3.5, max_value = 7.0)
    X6 = st.number_input("Orientation", min_value = 2.0, max_value = 5.0)
    X7 = st.number_input("Glazing Area", min_value = 0.0, max_value = 0.4)
    X8 = st.number_input("Glazing Area Distribution", min_value = 0.0, max_value = 5.0)



if st.button("Predict HL and CL"):
    input_data = np.array([[X1,X2,X3,X4,X5,X6,X7,X8]])
    scaled_input = scaler.transform(input_data)
    prediction_scaled = model.predict(scaled_input)
    prediction = np.round(scalerY.inverse_transform(prediction_scaled))
    st.success(f"Heating Load: {prediction[0][0]}, Cooling Load: {prediction[0][1]}")