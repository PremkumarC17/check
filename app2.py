import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Crop Prediction", page_icon=":guardsman:", initial_sidebar_state="auto")
#st.set_style("assets/css/style.css")
#st.set_theme('dark')

st.markdown("""
<style>
body {
    background-image: url(""C:/Users/PREMKUMAR C/Desktop/New folder (2)/tomasz-filipek-joOVC9d-jis-unsplash.jpg");
    background-color:red;
    background-size: cover;
    background-repeat: no-repeat;
}
</style>
<body>
</body>
""", unsafe_allow_html=True)



# load the dataset
data = pd.read_csv("crops_data.csv")
#X = data[['temperature', 'rainfall', 'humidity', 'croptype', 'soiltype', 'soilmoisture']]
X = data[['temperature', 'rainfall', 'humidity', 'soiltype', 'soilmoisture']]
y = data['crop']

# encode the categorical variables
encoder = LabelEncoder()
#X['croptype'] = encoder.fit_transform(X['croptype'])
X['soiltype'] = encoder.fit_transform(X['soiltype'])
y = encoder.fit_transform(y)


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# train the classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# rest of the code is same as before
# predict using the encoded labels
y_pred = clf.predict(X_test)

# convert the encoded labels back to original labels
y_pred = encoder.inverse_transform(y_pred)

# create the Streamlit app
st.title("Crop Prediction")

#get user input
temperature = st.number_input("Temperature (F)", min_value=0, max_value=120)
#rainfall = st.number_input("Rainfall (inches)", min_value=0.0)
rainfall = st.slider("Rainfall (inches)", min_value=0.0,max_value=30.0,step=0.1,value=1.2)
#humidity = st.number_input("Humidity (%)", min_value=0, max_value=100)
humidity =st.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1)
#croptype = st.selectbox("Crop Type", ['corn', 'wheat', 'soybeans', 'cotton','barley'])
soiltype = st.selectbox("Soil Type", ['clay', 'sandy', 'loamy'])
#soilmoisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100)
soilmoisture =st.slider("Soil Moisture (%)", min_value=0, max_value=100, value=50, step=1)

#convert the user input to numerical values
#matching the values in the dataset
#croptype = {'corn':0, 'wheat':1, 'soybeans':2, 'cotton':3 ,'barley':4}[croptype]
soiltype = {'clay':0, 'sandy':1, 'loamy':2}[soiltype]

#make the prediction
if st.button("Predict"):
    #result = clf.predict([[temperature, rainfall, humidity, croptype, soiltype, soilmoisture]])
    result = clf.predict([[temperature, rainfall, humidity, soiltype, soilmoisture]])
    st.balloons()
    #st.info("This is an example of a message with a balloon icon.")
    st.success("Suitable crop for this place is: {}".format(encoder.inverse_transform(result)[0]))
    if (encoder.inverse_transform(result)[0])=="barley":
        image_path = "barley.jpeg"
        st.image(image_path, caption='Barley', use_column_width=True)


st.line_chart(data[['temperature', 'humidity']])


