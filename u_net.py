from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization 
from tensorflow.keras.optimizers import Adam

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def modelUnet(num_classes = 2, input_shape= (1,800,608,3)):
    img_input = Input(input_shape)                                          

    # Block 1
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(img_input) # 800 608 32
    x = BatchNormalization(name='bn_0')(x)                                            # 
    x = Activation('relu',name='a_0')(x)                                              # 

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)         # 
    x = BatchNormalization(name='bn_1')(x)                                            # 
    block_1_out = Activation('relu',name='a_1')(x)                                    # 

    x = MaxPooling2D(name='m_0')(block_1_out)                                        # 400, 304, 32     Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)         # 400, 304, 64
    x = BatchNormalization(name='bn_2')(x)                                            # 
    x = Activation('relu',name='a_2')(x)                                              # 

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)         # 
    x = BatchNormalization(name='bn_3')(x)                                            # 
    block_2_out = Activation('relu',name='a_3')(x)                                    # 

    x = MaxPooling2D(name='m_1')(block_2_out)                                        # 200, 152, 64 

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)        # 200, 152, 128
    x = BatchNormalization(name='bn_4')(x)                                            # 
    x = Activation('relu',name='a_4')(x)                                              # 

    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)        # 200, 152, 128
    x = BatchNormalization(name='bn_5')(x)                                            # 
    block_3_out = Activation('relu',name='a_5')(x)                                    # 200, 152, 128

    x = MaxPooling2D(name='m_2')(block_3_out)                                        # 100, 76, 128

    # Block 4  
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)        # 100, 76, 256
    x = BatchNormalization(name='bn_6')(x)                                            # 
    x = Activation('relu',name='a_6')(x)                                              # 

    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)        # 100, 76, 256
    x = BatchNormalization(name='bn_7')(x)                                            # 
    block_4_out = Activation('relu',name='a_7')(x)                                    # block_4_out

    
    x = MaxPooling2D(name='m_3')(block_4_out)                                        # 50, 38, 256
#----------
    # Block 5  
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)        # 50, 38, 512
    x = BatchNormalization(name='bn_8')(x)                                             
    x = Activation('relu',name='a_8')(x)                                               

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)        # 50, 38, 512
    x = BatchNormalization(name='bn_9')(x)                                             
    block_5_out = Activation('relu',name='a_9')(x)                                    # block_5_out

    
    x = MaxPooling2D(name='m_4')(block_5_out)                                        # 25, 19, 512


    
    # Center
    x = Conv2D(512, (3, 3), padding='same', name='blockC_conv1')(x)        # 25, 19, 512
    x = BatchNormalization(name='bn_10')(x)                                            # 
    x = Activation('relu',name='a_10')(x)                                              # 

    x = Conv2D(512, (3, 3), padding='same', name='blockC_conv2')(x)        # 25, 19, 512
    x = BatchNormalization(name='bn_11')(x)                                            # 
    x = Activation('relu',name='a_11')(x) 


    # UP 5
    
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same',name='t_0')(x)    # 50, 38, 512  
    x = BatchNormalization(name='bn_12')(x)                                            # 
    x = Activation('relu',name='a_12')(x)                                              # 

    x = concatenate([x, block_5_out],name='c_0') 
    x = Conv2D(256, (3, 3), padding='same',name='co_0')(x)                             # 
    x = BatchNormalization(name='bn_13')(x)                                            # 
    x = Activation('relu',name='a_13')(x)                                              # 
 
    x = Conv2D(256, (3, 3), padding='same',name='co_01')(x)                             # 
    x = BatchNormalization(name='bn_14')(x)                                            # 
    x = Activation('relu',name='a_14')(x)                                              # 

#--------------------------------------------------------------------------------------------
    # UP 4
    
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',name='t_1')(x)    # 100, 76, 256  
    x = BatchNormalization(name='bn_15')(x)                                            # 
    x = Activation('relu',name='a_15')(x)                                              # 

    x = concatenate([x, block_4_out],name='c_1') 
    x = Conv2D(256, (3, 3), padding='same',name='co_2')(x)                             # 
    x = BatchNormalization(name='bn_16')(x)                                            # 
    x = Activation('relu',name='a_16')(x)                                              # 
 
    x = Conv2D(256, (3, 3), padding='same',name='co_3')(x)                             # 
    x = BatchNormalization(name='bn_17')(x)                                            # 
    x = Activation('relu',name='a_17')(x)                                              # 
    
    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',name='t_2')(x)    # 200, 152, 128
    x = BatchNormalization(name='bn_18')(x)                                            # 
    x = Activation('relu',name='a_18')(x)                                              # 

    x = concatenate([x, block_3_out],name='c_2') 
    x = Conv2D(128, (3, 3), padding='same',name='co_4')(x)                             # 200, 152, 128
    x = BatchNormalization(name='bn_19')(x)                                            # 
    x = Activation('relu',name='a_19')(x)                                              # 

    x = Conv2D(128, (3, 3), padding='same',name='co_5')(x)                             # 
    x = BatchNormalization(name='bn_20')(x)                                            # 
    x = Activation('relu',name='a_20')(x)                                              #      
    
    # UP 2
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',name='t_3')(x)     # 400, 304, 64
    x = BatchNormalization(name='bn_21')(x)                                            # 
    x = Activation('relu',name='a_21')(x)                                              # 

    x = concatenate([x, block_2_out],name='c_3') 
    x = Conv2D(64, (3, 3), padding='same',name='co_6')(x)                              # 
    x = BatchNormalization(name='bn_22')(x)                                            # 
    x = Activation('relu',name='a_22')(x)                                              # 

    x = Conv2D(64, (3, 3), padding='same',name='co_7')(x)                              # 
    x = BatchNormalization(name='bn_23')(x)                                            # 
    x = Activation('relu',name='a_23')(x)                                              # 
    
    # UP 1
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',name='t_4')(x)     # 800, 608, 32
    x = BatchNormalization(name='bn_24')(x)                                            # 
    x = Activation('relu',name='a_24')(x)                                              # 

    x = concatenate([x, block_1_out],name='c_4')
    x = Conv2D(32, (3, 3), padding='same',name='co_8')(x)                              # 
    x = BatchNormalization(name='bn_25')(x)                                            # 
    x = Activation('relu',name='an_25')(x)                                              # 

    x = Conv2D(32, (3, 3), padding='same',name='co_9')(x)                              # 
    x = BatchNormalization(name='bn_26')(x)                                            # 
    x = Activation('relu',name='a_26')(x)                                              # 

    x = Conv2D(num_classes,(3,3), activation='softmax', padding='same',name='co_10')(x) # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)                                            # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=0.0001), #1e-3
                  #loss='sparse_categorical_crossentropy',
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    
    return model
