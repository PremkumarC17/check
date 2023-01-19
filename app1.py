import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

st.title("Crop Prediction using Random Forest")

file_path = st.file_uploader("Upload your dataset in csv format", type=["csv"])
if file_path is not None:
    data = load_data(file_path)
    st.dataframe(data.head())
    data = pd.get_dummies(data, columns=["Soil_Type","Soil_Moisture"])

    X = data.drop(["Crop_Type"], axis=1)
    y = data["Crop_Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    st.write("Number of training samples:", X_train.shape[0])
    st.write("Number of testing samples:", X_test.shape[0])

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)

    if st.checkbox("Show feature importance"):
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        st.write(importances.sort_values(ascending=False))

else:
    st.error("Please upload a csv file")
