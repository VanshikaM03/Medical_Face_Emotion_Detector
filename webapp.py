import base64
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type:ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore


# Load your pre-trained emotion detection model
# Ensure you have the model file in your directory
model = load_model('D:/Face_emotion_detection/emotiondetector2.h5')

# Define function to predict emotion
# Change this to the link of your image or local path


# Define your pages
pages = {
    "Home": "Display the home page",
    "Monitor": "Monitor patient data",

}

# Let user choose a page
page = st.sidebar.radio("Choose a page:", list(pages.keys()))


def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


background_image_base64 = get_base64_of_image(
    "D:/Face_emotion_detection/Untitleddesign.jpg")


def add_bg_image(base64_string):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_string}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


if page == "Home":
    add_bg_image(background_image_base64)


def predict_emotion(img):
    emotions = ['Angry', 'Disgust', 'Fear',
                'Happy', 'Neutral', 'Sad', 'Surprise']
    # Resize image to match model's expected input size
    img = cv2.resize(img, (48, 48))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Model expects pixels to be in [0, 1]
    prediction = model.predict(img)
    max_index = np.argmax(prediction[0])
    return emotions[max_index]


# Streamlit app code
st.header('Monitor Your Patient')
patient_history = {
    "Name": "John Doe",
    "Age": 34,
    "Condition": "Coma",
    "Admission Date": "2024-04-01",
    "Medical History": [
        {"Date": "2023-10-15", "Condition": "Migraine"},
        {"Date": "2023-12-20", "Condition": "High Fever"},
        {"Date": "2024-01-30", "Condition": "Minor Seizure"},
        {"Date": "2024-04-01", "Condition": "Coma, post-traumatic"}
    ]
}
# Sidebar
doctor_name = "Dr. John Doe"
qualifications = ["MD from University XYZ",
                  "Specialist in Internal Medicine", "10 years of experience"]
st.sidebar.markdown(f"**{doctor_name}**")
st.sidebar.markdown("### Qualifications")
for qualification in qualifications:
    st.sidebar.markdown(f"- {qualification}")
st.sidebar.header('Navigation')
option = st.sidebar.selectbox(
    'Select an Option', ('Home', 'Upload Photo', 'View Patient Data', 'Add Patient Data'))

if option == 'Home':
    st.subheader('Welcome to the Patient Monitoring Dashboard')
    st.write("""
    ### OpenCV Emotion Detection Project Overview
    This application leverages advanced facial recognition and emotion detection technologies to assist medical professionals in monitoring patients. The emotion detection system uses OpenCV and deep learning models to analyze facial expressions in real-time.

    ### Application in Medical Monitoring
    In medical settings, particularly for patients in non-responsive states such as coma, traditional vital signs may not provide a complete picture of the patient's immediate well-being. Our system allows for continuous observation of a patient's facial expressions, enabling doctors to detect subtle changes that might indicate pain, distress, or improvements in the patient's condition.

    This tool is particularly useful as attendants cannot be present at all times. By automating the detection of emotional responses, caregivers can be alerted to changes in the patient's condition that might otherwise go unnoticed. This could facilitate timely medical interventions, potentially improving patient outcomes.
    """)
elif option == 'Upload Photo':
    st.subheader('Upload a Photo')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = np.array(load_img(uploaded_file, color_mode='grayscale'))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        if st.button('Detect Emotion'):
            emotion = predict_emotion(image)
            st.write(f"Detected Emotion: {emotion}")
elif option == 'View Patient Data':
    st.subheader('Patient Data')
    # Display patient history
    st.write(f"**Name:** {patient_history['Name']}")
    st.write(f"**Age:** {patient_history['Age']}")
    st.write(f"**Current Condition:** {patient_history['Condition']}")
    st.write(f"**Admission Date:** {patient_history['Admission Date']}")
    st.subheader('Medical History')
    for event in patient_history['Medical History']:
        st.write(f"{event['Date']}: {event['Condition']}")
elif option == 'Add Patient Data':
    st.subheader('Add New Data')
