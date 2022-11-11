import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

#from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
#--------------------------------------------------

from tensorflow.keras import utils

import u_net

img_width = 192
img_height = 256
num_classes = 2

def index2color(ind):
    index = np.argmax(ind) # Получаем индекс максимального элемента
    color = index*255
    return color # Возвращаем цвет пикслея

model = u_net.modelUnet(num_classes,(img_height,img_width, 3))
model.load_weights('model_weights_P.h5')
#--------------------------------------------------
def preprocess_image(img):
    img = img.resize((192, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x
    
n_classes = 2
def Prediction(i):
    pr = np.array(model.predict(i)) # Предиктим картинку
    pr = pr.reshape(-1, 2) # Решейпим предикт
    pr1 = [] # Пустой лист под сегментированную картинку из predicta
    for q in pr: 
       pr1.append(index2color(q)) # Переводим индекс в писксель
    pr1 = np.array(pr1)
    pr1 = pr1.reshape(img_height,img_width,1)
    return pr1
    
def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        
        img = Image.open(io.BytesIO(image_data))
        result = st.button('Распознать изображение')
        if result:
            x = preprocess_image(img)
            pred_ar = Prediction(x)
            pred_im = Image.fromarray(pred)
            #st.image(pred_im)
            st.text(pred_ar.shape)
            st.text(x.shape)
        return image_data
    else:
        return None

st.title('Загрузка, скачивание изображений')

s = load_image()



if s is not None:
    st.download_button(label='скачать',data=s,file_name = 'O.jpg')
    
