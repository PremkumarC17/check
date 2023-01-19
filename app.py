import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv("crop_data.csv")

# Split the data into training and testing sets
X = data.drop("yield", axis=1)
y = data["yield"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the random forest model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Define the Streamlit app
#st.set_title("Crop Yield Prediction")
st.set_page_config(page_title="Stock Price", layout="centered", initial_sidebar_state = "auto")
st.write("Enter the following information to predict the yield of a crop:")

# Get input from the user
soil_quality = st.number_input("Soil Quality (0-10)", min_value=0, max_value=10)
temperature = st.number_input("Temperature (F)", min_value=-40, max_value=120)
precipitation = st.number_input("Precipitation (inches)", min_value=0)

# Use the model to make a prediction
prediction = rf.predict([[soil_quality, temperature, precipitation]])

# Display the prediction
st.write("Predicted yield:", prediction[0])

# Calculate and display the model's performance on the test set
test_predictions = rf.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
st.write("Model performance:", mae)
