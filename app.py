import streamlit as st 
import pickle 
import numpy as np 
import pandas as pd 

with open('xgb_model_final.pkl','rb') as file:
    model = pickle.load(file)

st.title('Cab services')


import streamlit as st

# Slider to select the range for Pickup Longitude
pickup_longitude = st.slider('Select Pickup Longitude:', min_value=-74.25219, max_value=-72.986534, step=0.1, key="pickup_longitude")

# Slider to select the range for Pickup Latitude
pickup_latitude = st.slider('Select Pickup Latitude:', min_value=40.573143, max_value=41.709557	, step=0.1, key="pickup_latitude")

# Slider to select the range for Dropoff Longitude
dropoff_longitude = st.slider('Select Dropoff Longitude:', min_value=-74.263245, max_value=-72.990967, step=0.1, key="dropoff_longitude")

# Slider to select the range for Dropoff Latitude
dropoff_latitude = st.slider('Select Dropoff Latitude:', min_value=40.568973, max_value=41.696683, step=0.1, key="dropoff_latitude")

# Example for a passenger count range
passenger_count = st.slider('Select Passenger Count:', min_value=1, max_value=6, step=1, key="passenger_count")

pickup_datetime_weekday = st.slider('Select pickup_datetime_weekday Count:', min_value=1, max_value=7, step=1, key="pickup_datetime_weekday")

# for calculating the median values from training data
median_values = {
    'pickup_datetime_year': 2012.0,
    'pickup_datetime_month': 6.0,
    'pickup_datetime_day': 16.0,
    'pickup_datetime_hour':14.0,
    'trip_distance':2.1532199782051693,
    'jfk_drop_distance': 21.17354668855731,
    'lga_drop_distance': 9.5208179941764,
    'ewr_drop_distance': 17.95877286271942,
    'met_drop_distance': 3.720924841580352,
    'wtc_drop_distance': 5.4886501689597775,
}
customer_input = np.array([[pickup_longitude, pickup_latitude,
                            dropoff_longitude, dropoff_latitude,
                            passenger_count,pickup_datetime_weekday]])

missing_features = np.array([median_values[key] for key in median_values])


if st.button('Predict'):
    input_data = np.concatenate([customer_input, missing_features.reshape(1, -1)], axis=1)
    prediction = model.predict(input_data)
    st.write(f'Prediction: ${prediction[0]:.2f}')