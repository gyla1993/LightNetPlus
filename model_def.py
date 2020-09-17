# This file defines structure of all models.

from keras.models import Model
from keras.layers import Input, Conv2D, TimeDistributed, ConvLSTM2D, Lambda, Conv3D, Cropping2D,\
                                    Concatenate, Reshape, Activation, Cropping3D, Conv2DTranspose,\
                                    Maximum, Add, Multiply, Subtract, Permute
import keras.backend as K
import numpy as np
from global_var import dim_WRF, dim_AWS, num_LIG, num_AWS, num_PRED, use_good_start



def LightNetPlus_WRF():
    # Remove the initial observation from the final_states

    encoder1_inputs = Input(shape=(None, 159, 159, dim_WRF), name='encoder1_inputs')  # (bs, 6, 159, 159, dim_WRF)
    encoder1_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_1')(encoder1_inputs)
    encoder1_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_2')(encoder1_conv2d_1)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=False,
                                           name='en1_convlstm')(encoder1_conv2d_2)
    # --------------------------------------------------------------------------------
    filters_list = [64, 0, 0]
    h1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h = h1
    c = c1
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)


    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)  # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model(inputs=[encoder1_inputs, decoder_inputs_], outputs=[decoder_outputs],
                 name='LightNetPlus_WRF')

def LightNetPlus_LIG():
    # Remove the initial observation from the final_states


    # encoder2: layers definition && data flow  --------------------------------------
    encoder2_inputs = Input(shape=(None, 159, 159, 1), name='encoder2_inputs')  # (bs, 3, 159, 159, 1)
    encoder2_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_1')(encoder2_inputs)
    encoder2_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_2')(encoder2_conv2d_1)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=False,
                                           name='en2_convlstm')(encoder2_conv2d_2)

    filters_list = [0, 64, 0]
    h2 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)

    h = h2
    c = c2
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)  # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model(inputs=[encoder2_inputs, decoder_inputs_], outputs=[decoder_outputs],
                 name='LightNetPlus_LIG')

def LightNetPlus_AWS():
    # Remove the initial observation from the final_states

    # encoder3: layers definition && data flow  --------------------------------------
    encoder3_inputs = Input(shape=(None, 159, 159, dim_AWS), name='encoder3_inputs')
    encoder3_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_1')(encoder3_inputs)
    encoder3_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_2')(encoder3_conv2d_1)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en3_convlstm')(encoder3_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  --------------------

    filters_list = [0, 0, 64]
    h3 = Conv2D(filters=filters_list[2], kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=filters_list[2], kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = h3
    c = c3
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)  # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model(inputs=[encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs],
                 name='LightNetPlus_AWS')

def LightNetPlus_WRF_LIG():
    # Remove the initial observation from the final_states

    encoder1_inputs = Input(shape=(None, 159, 159, dim_WRF), name='encoder1_inputs')  # (bs, 6, 159, 159, dim_WRF)
    encoder1_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_1')(encoder1_inputs)
    encoder1_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_2')(encoder1_conv2d_1)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=False,
                                           name='en1_convlstm')(encoder1_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder2: layers definition && data flow  --------------------------------------
    encoder2_inputs = Input(shape=(None, 159, 159, 1), name='encoder2_inputs')  # (bs, 3, 159, 159, 1)
    encoder2_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_1')(encoder2_inputs)
    encoder2_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_2')(encoder2_conv2d_1)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=False,
                                           name='en2_convlstm')(encoder2_conv2d_2)

    # encoder to decoder: layers definition && data flow  --------------------

    filters_list = [40, 24, 0]
    h1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h = Concatenate(axis=-1)([h1, h2])  # (bs,  40, 40, 48+16=64)
    c = Concatenate(axis=-1)([c1, c2])  # (bs,  40, 40, 48+16=64)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)  # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model(inputs=[encoder1_inputs, encoder2_inputs, decoder_inputs_], outputs=[decoder_outputs],
                 name='LightNetPlus_WRF_LIG')

def LightNetPlus_WRF_AWS():
    # Remove the initial observation from the final_states

    encoder1_inputs = Input(shape=(None, 159, 159, dim_WRF), name='encoder1_inputs')  # (bs, 6, 159, 159, dim_WRF)
    encoder1_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_1')(encoder1_inputs)
    encoder1_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_2')(encoder1_conv2d_1)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=False,
                                           name='en1_convlstm')(encoder1_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder3: layers definition && data flow  --------------------------------------
    encoder3_inputs = Input(shape=(None, 159, 159, dim_AWS), name='encoder3_inputs')
    encoder3_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_1')(encoder3_inputs)
    encoder3_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_2')(encoder3_conv2d_1)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en3_convlstm')(encoder3_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  --------------------

    filters_list = [40, 0, 24]
    h1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=filters_list[0], kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h3 = Conv2D(filters=filters_list[2], kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=filters_list[2], kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h3])  # (bs,  40, 40, 48+16=64)
    c = Concatenate(axis=-1)([c1, c3])  # (bs,  40, 40, 48+16=64)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)  # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model(inputs=[encoder1_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs],
                 name='LightNetPlus_WRF_AWS')

def LightNetPlus_LIG_AWS():
    # Remove the initial observation from the final_states

    # encoder2: layers definition && data flow  --------------------------------------
    encoder2_inputs = Input(shape=(None, 159, 159, 1), name='encoder2_inputs')  # (bs, 3, 159, 159, 1)
    encoder2_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_1')(encoder2_inputs)
    encoder2_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en2_conv2d_2')(encoder2_conv2d_1)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=False,
                                           name='en2_convlstm')(encoder2_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder3: layers definition && data flow  --------------------------------------
    encoder3_inputs = Input(shape=(None, 159, 159, dim_AWS), name='encoder3_inputs')
    encoder3_conv2d_1 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_1')(encoder3_inputs)
    encoder3_conv2d_2 = TimeDistributed(
        Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en3_conv2d_2')(encoder3_conv2d_1)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en3_convlstm')(encoder3_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  --------------------

    filters_list = [0, 32, 32]
    h2 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=filters_list[1], kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=filters_list[2], kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=filters_list[2], kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h2, h3])  # (bs,  40, 40, 48+16=64)
    c = Concatenate(axis=-1)([c2, c3])  # (bs,  40, 40, 48+16=64)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------
    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)  # (bs, 6, 159, 159, 1)
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model(inputs=[encoder2_inputs, encoder3_inputs, decoder_inputs_], outputs=[decoder_outputs],
                 name='LightNetPlus_LIG_AWS')

def LightNetPlus(): # encoder downsample twice, first convolute then concat
    encoder1_inputs = Input(shape=(None, 159, 159, dim_WRF), name='encoder1_inputs')  # (bs, 6, 159, 159, dim_WRF)
    # print('encoder1_inputs:', K.shape(encoder1_inputs))
    encoder1_conv2d_1 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_1')(encoder1_inputs)
    encoder1_conv2d_2 = TimeDistributed(
        Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='en1_conv2d_2')(encoder1_conv2d_1)
    encoder1_convlstm, h1, c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en1_convlstm')(encoder1_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder2: layers definition && data flow  --------------------------------------
    encoder2_inputs = Input(shape=(None, 159, 159, 1), name='encoder2_inputs')  # (bs, 3, 159, 159, 1)
    # print('encoder2_inputs:', K.shape(encoder2_inputs))
    encoder2_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en2_conv2d_1')(encoder2_inputs)
    encoder2_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en2_conv2d_2')(encoder2_conv2d_1)
    encoder2_convlstm, h2, c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en2_convlstm')(encoder2_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder3: layers definition && data flow  --------------------------------------
    encoder3_inputs = Input(shape=(None, 159, 159, dim_AWS), name='encoder3_inputs')
    # print('encoder3_inputs:', K.shape(encoder3_inputs))
    encoder3_conv2d_1 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en3_conv2d_1')(encoder3_inputs)
    encoder3_conv2d_2 = TimeDistributed(Conv2D(filters=4, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
                                      name='en3_conv2d_2')(encoder3_conv2d_1)
    encoder3_convlstm, h3, c3 = ConvLSTM2D(filters=32, kernel_size=(5, 5),
                                           return_state=True, padding='same', return_sequences=True,
                                           name='en3_convlstm')(encoder3_conv2d_2)
    # --------------------------------------------------------------------------------
    # encoder to decoder: layers definition && data flow  --------------------

    h1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='h1_conv2d', activation='relu')(h1)
    c1 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", name='c1_conv2d', activation='relu')(c1)
    h2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h2_conv2d', activation='relu')(h2)
    c2 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c2_conv2d', activation='relu')(c2)
    h3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='h3_conv2d', activation='relu')(h3)
    c3 = Conv2D(filters=16, kernel_size=(1, 1), padding="same", name='c3_conv2d', activation='relu')(c3)
    h = Concatenate(axis=-1)([h1, h2, h3])  # (bs,  40, 40, 48+16=64)
    c = Concatenate(axis=-1)([c1, c2, c3])  # (bs,  40, 40, 48+16=64)
    # --------------------------------------------------------------------------------

    # decoder: layers definition -------------------------------

    decoder_inputs_ = Input(shape=(None, 159, 159, 1), name='decoder_inputs')
    decoder_inputs = decoder_inputs_
    if use_good_start:
        decoder_inputs = Lambda(lambda x: np.power(x, 1. / 3) * 4 - 3.5)(decoder_inputs)
        sigmoid = Activation('sigmoid')
        decoder_inputs = sigmoid(decoder_inputs)

    de_conv2d_1 = TimeDistributed(
        Conv2D(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_1')
    de_conv2d_2 = TimeDistributed(
        Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'), name='de_conv2d_2')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm', padding='same', return_state=True,
                             return_sequences=True)
    de_conv2dT_1 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_1')
    de_conv2dT_2 = TimeDistributed(
        Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        name='de_conv2dT_2')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding='same'),
                                    name='de_out_conv2d')
    # ----------------------------------------------------------
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))
    # decoder: data flow-----------------------------------------

    decoder_outputs_list = []
    for i in range(num_PRED):
        decoder_conv2d_1 = de_conv2d_1(decoder_inputs)
        decoder_conv2d_2 = de_conv2d_2(decoder_conv2d_1)
        decoder_convlstm, h, c = de_convlstm([decoder_conv2d_2, h, c])
        decoder_conv2dT_1 = de_conv2dT_1(decoder_convlstm)
        decoder_conv2dT_2 = de_conv2dT_2(decoder_conv2dT_1)
        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT_2)  # (bs, 1, 160, 160, 1)
        decoder_output = cropper(decoder_out_conv2d)  # (bs, 1, 159, 159, 1)
        decoder_outputs_list.append(decoder_output)
        if use_good_start:
            decoder_output = sigmoid(decoder_output)
            decoder_inputs = decoder_output

    decoder_outputs = Concatenate(axis=1)(decoder_outputs_list)   # (bs, 6, 159, 159, 1)
    # print('decoder_outputs:', K.shape(decoder_outputs))
    decoder_outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(decoder_outputs)
    # ----------------------------------------------------------
    return Model([encoder1_inputs, encoder2_inputs, encoder3_inputs, decoder_inputs_], decoder_outputs, name='LightNetPlus')

def StepDeep():
    WRF_inputs = Input(shape=(num_PRED, 159, 159, dim_WRF))   # (bs, 6, 159, 159, dim_WRF)
    _history_inputs = Input(shape=(num_LIG, 159, 159, 1))  # (bs,3,159,159,1)
    # history_inputs = Lambda(lambda x: K.squeeze(x, axis=-1))(_history_inputs)   # (bs, 3, 159, 159)
    history_inputs = Permute((4, 2, 3, 1))(_history_inputs)              #  (bs, 1, 159, 159, 3)
    _sta_inputs = Input(shape=(num_AWS, 159, 159, dim_AWS), name='sta_inputs')
    sta_inputs = Permute((2, 3, 1, 4))(_sta_inputs)
    sta_inputs = Reshape((1, 159, 159, dim_AWS*num_AWS))(sta_inputs)

    conv_1 = Conv3D(filters=128, kernel_size=(2, 1, 1), padding='same', name='conv3d_1')(WRF_inputs)
    conv_1 = Activation('relu')(conv_1)
    conv_2 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', name='conv3d_2')(conv_1)
    conv_2 = Activation('relu')(conv_2)
    conv_3 = Conv3D(filters=256, kernel_size=(2, 3, 3), padding='same', name='conv3d_3')(conv_2)
    conv_3 = Activation('relu')(conv_3)
    conv_4 = Conv3D(filters=128, kernel_size=(3, 1, 1), padding='same', name='conv3d_4')(conv_3)
    conv_4 = Activation('relu')(conv_4)
    conv_5 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', name='conv3d_5')(conv_4)
    conv_5 = Activation('relu')(conv_5)
    conv_6 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', name='conv3d_6')(conv_5)
    conv_6 = Activation('relu')(conv_6)
    steps = []
    for i in range(num_PRED):
        conv_6_i = Cropping3D(cropping=((i, num_PRED - i - 1), (0, 0), (0, 0)))(conv_6)   # (bs, 1, 159, 159, 64)
        conv2d_in = Concatenate(axis=-1)([history_inputs, conv_6_i, sta_inputs])                        # (bs, 1, 159, 159, 64+3)
        conv2d_in = Lambda(lambda x: K.squeeze(x, axis=1))(conv2d_in)  # (bs, 159, 159, 67)
        conv2d_1_i = Conv2D(filters=64, kernel_size=(7, 7), padding='same', name='conv2d_1_%d' % i)(conv2d_in)
        conv2d_1_i = Activation('relu')(conv2d_1_i)
        conv2d_2_i = Conv2D(filters=1, kernel_size=(7, 7), padding='same', name='conv2d_2_%d' % i)(conv2d_1_i)
        steps.append(conv2d_2_i)
    conv_out = Concatenate(axis=1)(steps)  # (bs, 6, 159, 159, 1)
    outputs = Reshape((-1, 159 * 159, 1), input_shape=(-1, 159, 159, 1))(conv_out)
    return Model([WRF_inputs, _history_inputs, _sta_inputs], outputs, name='StepDeep')



