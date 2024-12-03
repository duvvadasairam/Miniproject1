import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS

# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/2.jpg')

# Streamlit app title
st.title("Project : A Lesion-Based Diabetic Retinopathy Detection  Through Hybrid Deep Learning Model")

# Load the model
model = load_model("model.h5")

# Define image dimensions and categories
WIDTH, HEIGHT = 65, 65
categories = ['Mild','Moderate','No_DR','Proliferate_DR','Severe']

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    image = np.array(image)
    
    # Ensure the image has 3 channels (RGB)
    if image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    test_image = cv2.resize(image, (WIDTH, HEIGHT))
    test_data = np.array(test_image, dtype="float") / 255.0
    test_data = test_data.reshape([-1, WIDTH, HEIGHT, 3])
    return image, test_data

# Function to segment the image using thresholding
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    segmented_image = cv2.bitwise_and(image, image, mask=thresholded)
    return segmented_image

# Streamlit interface
st.title("Image Classification Prediction")
st.write("Upload images to get the predictions.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    predictions_list = []  # Initialize an empty list to store predictions
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Load image with PIL
            image = Image.open(uploaded_file)
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            test_image_o, test_data = load_and_preprocess_image(image)
            
            # Make prediction
            pred = model.predict(test_data)
            predictions = np.argmax(pred, axis=1)  # return to label
            
            # Append prediction to the list
            predictions_list.append(categories[predictions[0]])
            
            # Display the prediction
            st.write(f'Prediction: {categories[predictions[0]]}')
            
            # Display the image with the prediction title
            fig = plt.figure()
            fig.patch.set_facecolor('xkcd:white')
            plt.title(categories[predictions[0]])
            plt.imshow(cv2.cvtColor(test_image_o, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axes
            st.pyplot(fig)
            
            # Segment the image
            segmented_image = segment_image(test_image_o)
            
            # Display the segmented image
            fig = plt.figure()
            fig.patch.set_facecolor('xkcd:white')
            plt.title('Segmented Image')
            plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axes
            st.pyplot(fig)
    
    # Display all predictions
    st.write("All Predictions:")
    st.write(predictions_list)
    
    # Convert predictions list to string and create audio
    predictions_text = ' '.join(predictions_list)
    language = 'en'
    speech = gTTS(text=predictions_text, lang=language, slow=False)
    speech.save("sample.mp3")
    audio_path = "sample.mp3"  # Replace with the path to your MP3 audio file

    st.audio(audio_path, format='audio/mp3')
else:
    st.write("Please upload image files.")
