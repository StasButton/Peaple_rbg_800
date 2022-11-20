from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization 
from tensorflow.keras.optimizers import Adam

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def modelUnet(num_classes = 2, input_shape= (1,800,608,3)):
    img_input = Input(input_shape)                                         

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input) # 800 608 32
    x = BatchNormalization()(x)                                           
    x = Activation('relu')(x)                                              

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)         
    x = BatchNormalization()(x)                                            
    block_1_out = Activation('relu')(x)                                    # 800 608 32                                   

    x = MaxPooling2D()(block_1_out)                                        # 400, 304, 32     

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)         
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)         
    x = BatchNormalization()(x)                                            
    block_2_out = Activation('relu')(x)                                    # 400, 304, 32                                   

    x = MaxPooling2D()(block_2_out)                                        # 200, 152, 64 

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)        
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)        
    x = BatchNormalization()(x)                                            
    block_3_out = Activation('relu')(x)                                    # 200, 152, 64

    x = MaxPooling2D()(block_3_out)                                        # 100, 76 ,128

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)         
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)         
    x = BatchNormalization()(x)                                            
    block_4_out = Activation('relu')(x)                                     # 100, 76 ,128

    x = MaxPooling2D()(block_4_out)                                         # 50, 36, 256

    # Center
    x = Conv2D(256, (3, 3), padding='same', name='blockC_conv1')(x)        
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(256, (3, 3), padding='same', name='blockC_conv2')(x)         
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)  

    # UP 4
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)    #  100, 76, 256   
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = concatenate([x, block_4_out])                                      #  100, 76, 256   |   100, 76, 256
    x = Conv2D(256, (3, 3), padding='same')(x)                            
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              
 
    x = Conv2D(256, (3, 3), padding='same')(x)                              
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              
    
    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)    # 200, 152, 128  
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = concatenate([x, block_3_out])                                       # 200, 152, 128 |  # 200, 152, 128 
    x = Conv2D(128, (3, 3), padding='same')(x)                             
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(128, (3, 3), padding='same')(x)                             
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              
          
    
    # UP 2
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)     # 400, 304, 64 
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = concatenate([x, block_2_out])                                       # 400, 304, 64 |  # 400, 304, 64 
    x = Conv2D(64, (3, 3), padding='same')(x)                              
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(64, (3, 3), padding='same')(x)                              
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              
    
    # UP 1
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)     # 800 608 32
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = concatenate([x, block_1_out])                                      # 800 608 32 | # 800 608 32
    x = Conv2D(32, (3, 3), padding='same')(x)                              
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(32, (3, 3), padding='same')(x)                              
    x = BatchNormalization()(x)                                            
    x = Activation('relu')(x)                                              

    x = Conv2D(num_classes,(3,3), activation='softmax', padding='same')(x) #  softmax

    model = Model(img_input, x)                                           

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  #loss='sparse_categorical_crossentropy',
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    
    return model
'''
#++++++++++++++++++++++++++++++++++++++++++++++ 
def index2color(ind):
    index = np.argmax(ind) # Получаем индекс максимального элемента
    color = index*255
    return color # Возвращаем цвет пикслея



def preprocess_image(img):
    img = img.resize((192, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def pedict2(fg,bg):
    pr = np.array(model.predict(fg)) # Предиктим картинку
    pr = pr.reshape(-1, 2) # Решейпим предикт
    fg = fg.reshape(-1, 3)
    for i , q in enumerate(pr): #start =1
        if np.argmax(q) > 0.5:
            bg[i] = fg[i]
    bg = bg.reshape(img_height,img_width,3)
    return bg

def bgload():
    uploaded_file = st.file_uploader(label='Выберите фон')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
#++++++++++++++++++++++++++++++++++++++++++++++ 
'''
