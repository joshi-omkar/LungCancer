import streamlit as st
from utility import utils, cnn_model

IMAGE_SIZE_UI = (200, 200)
IMAGE_SIZE_PREPROCESS = 64
LEARN_RATE = 1.0e-4
CH = 3
st.title('Lung Cancer Detection Using CNN Model')
config_dict = utils.get_config()

uploaded_file = st.file_uploader("Choose a Image")
if uploaded_file is not None:
    utils.pre_process_image(uploaded_file, IMAGE_SIZE_UI)
    if st.button('Do CNN Prediction'):
        utils.do_cnn_prediction(uploaded_file, IMAGE_SIZE_PREPROCESS, config_dict)





