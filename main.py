import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
#--------------------------------------------------

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import utils 

#import os 
#import cv2
#--------------------------------------------------

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        s = Image.open(io.BytesIO(image_data))
        return image_data 
    else:
        return None

st.title('Загрузка, скачивание изображений')
s = load_image()
if s is not None:
    st.download_button(label='скачать',data=s,file_name = 'O.jpg')



