import json
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time

from keras.models import load_model


def get_config():
    f = open('configs/config.json')
    return json.load(f)


def pre_process_image(uploaded_image, IMAGE_SIZE):
    #convert grey
    image = Image.open(uploaded_image)
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    input_image = Image.fromarray(image)

    gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    x1 = int(image.shape[0])
    y1 = int(image.shape[1])

    gs = cv2.resize(gs, (x1, y1))
    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print(threshold)

    grey_image = Image.fromarray(gs)

    black_and_white_image = Image.fromarray(threshold)

    col1, col2, col3 = st.columns(3)

    with col1:
       st.image(input_image, caption='Uploaded Image')

    with col2:
       st.image(grey_image, caption='Processed Gray Image')

    with col3:
       st.image(black_and_white_image, caption='Processed B/W Image')


def do_cnn_prediction(uploaded_image, IMAGE_SIZE, config_dict):
    # Model Architecture and Compilation
    start = time.time()

    model = load_model(config_dict["model_path"])

    image = Image.open(uploaded_image)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)

    img = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    img = img.astype('float32')
    img = img / 255.0
    print('img shape:', img)
    prediction = model.predict(img)
    print(np.argmax(prediction))
    plant = np.argmax(prediction)
    print(plant)

    cancer_type = None
    if plant == 0:
        cancer_type = "Bengin case"
    elif plant == 1:
        cancer_type = "Malignant case"
    elif plant == 2:
        cancer_type = "Normal case"

    if cancer_type is not None:
        end = time.time()
        duration = end - start
        st.write(f"We Detected Cancer type as :blue[{cancer_type}]")
        st.write(f"Read more about that cancer type")
        st.write(f"Total Duration for prediction is :blue[{duration} millisecond]")



