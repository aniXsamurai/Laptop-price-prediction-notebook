import pickle
import streamlit as st
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of Laptop
type_ = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM in GB', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input("Enter laptop weight (in kg):", min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])

# IPS
ips = st.selectbox('IPS Display', ['Yes', 'No'])

# Screensize
screen_size = st.number_input('Screen size (in inches):', min_value=10.0, max_value=20.0, step=0.1)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080','1366x768','1600x900','3840x2160',
    '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
])

# CPU
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# GHz
gHz = st.selectbox('Clock Speed (GHz)', [
    2.3,1.8,2.5,2.7,3.1,3.0,2.2,1.6,2.0,2.8,1.2,2.9,
    2.4,1.44,1.5,1.9,1.1,1.3,2.6,3.6,3.2,1.0,2.1,0.9,1.92
])

# Storage
hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])

# GPU
gpu = st.selectbox('GPU Brand', df['Gpu_brand'].unique())

# OS
os = st.selectbox('Operating System', df['os'].unique())

# Predict button
if st.button('Predict Price'):
    if screen_size == 0:
        st.warning("Please enter a valid screen size!")
    else:
        # Convert categorical inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Final query array
        query = np.array([company, type_, ram, weight, touchscreen, ips, ppi,
                          cpu, gHz, hdd, ssd, gpu, os], dtype=object).reshape(1, 13)

        # Predict
        predicted_price = np.exp(pipe.predict(query)[0])  # assuming model trained on log(price)
        st.success(f"The predicted price of this configuration is ₹{int(predicted_price):,}")
