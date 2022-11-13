import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
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

#++++++++++++++++++++++++++++++++++++++++++++++
def pedict2(fg,bg):
    pr = np.array(model.predict(fg)) # Предиктим картинку
    pr = pr.reshape(-1, 2) # Решейпим предикт
    fg = fg.reshape(-1, 3)
    for i , q in enumerate(pr): #start =1
        if np.argmax(q) > 0.5:
            bg[i] = fg[i]
    bg = bg.reshape(img_height,img_width,3)
    return bg
#++++++++++++++++++++++++++++++++++++++++++++++++ 
def bgload():
    uploaded_file = st.file_uploader(label='Выберите фон')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        
def load_result(ar):
    im  = utils.array_to_img(ar)
    st.image(im)
    pred_ar_int = ar.astype(np.uint8)
    im = Image.fromarray(pred_ar_int)
    return im          
    
def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        img = Image.open(io.BytesIO(image_data))
        x = preprocess_image(img)
        
        uploaded_file_bg = st.file_uploader(label='Выберите фон')
        if uploaded_file_bg is not None:
            image_data_bg = uploaded_file_bg.getvalue()
            st.image(image_data_bg)
            img_bg = Image.open(io.BytesIO(image_data_bg))
            x_bg = preprocess_image(img_bg)
            x_bg = x_bg.reshape(-1, 3)

            result = st.button('Заменить фон')
            if result:
                pred_ar = pedict2(x,x_bg) 
                im = load_result(pred_ar)
                
            if result:
            with io.BytesIO() as f:
                 im.save(f, format='JPEG')
                 data = f.getvalue()
            st.download_button(label='Скачать',data=data,file_name='change_bg.jpg')
                
                
            
        #return  sd
    else:
        return None
    
st.title('Замена фона на фотографиях людей')

load_image()

