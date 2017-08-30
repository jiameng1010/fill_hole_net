from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D, UpSampling2D, multiply, core
from keras.layers import Input, Conv2DTranspose, concatenate, Activation, Dense
from keras.layers.merge import Multiply, Add
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

def model_init(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    upconv5_c = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5_c, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1)

    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1])

    return model

def discriminator(input_shape):

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)
    pr1 = Dense(256, activation='relu')(conv6b)
    pr2 = Dense(1, activation='softmax')(conv6b)

    model = Model(input=a, outputs=pr2)

    return model

def model_init_binary(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    upconv5_c = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(conv6b)
    #pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5_c, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv5)
    #pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv4)
    #pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv3)
    #pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv2)
    #pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv1)
    #pr1b = Activation(K.softmax)(pr1)


    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1])

    return model

def model_judgement(input_shape):
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    close = Input(shape=(224, 320, 1))
    far = Input(shape=(224, 320, 1))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv6a)

    upconv5_c = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2),
                              input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(conv6b)
    # pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2, 2))(pr6)
    inter5 = concatenate([upconv5_c, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv5)
    # pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2, 2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv4)
    # pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2, 2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv3)
    # pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2, 2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv2)
    # pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2, 2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv1)
    # pr1b = Activation(K.softmax)(pr1)

    far_w = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_w = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)

    far_ww = Multiply()([far_w, far])
    close_ww = Multiply()([close_w, close])
    pre = Add()([far_ww, close_ww])
    #pre = core.Lambda(judgement_merge(pr1[:,:,:,1], far, pr1[:,:,:,2], close), output_shape=(1,))
    #pre = MyLayer()([pr1[:,:,:,1], far, pr1[:,:,:,2], close])

    model = Model(inputs=[a, far, close], outputs=pre)

    return model



def model_judgement2():
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    close1 = Input(shape=(224, 320, 1))
    far1 = Input(shape=(224, 320, 1))
    close2 = Input(shape=(112, 160, 1))
    far2 = Input(shape=(112, 160, 1))
    close3 = Input(shape=(56, 80, 1))
    far3 = Input(shape=(56, 80, 1))
    close4 = Input(shape=(28, 40, 1))
    far4 = Input(shape=(28, 40, 1))
    close5 = Input(shape=(14, 20, 1))
    far5 = Input(shape=(14, 20, 1))
    close6 = Input(shape=(7, 10, 1))
    far6 = Input(shape=(7, 10, 1))

    a = Input(shape=(448, 640, 6))
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv6a)

    upconv5_c = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2),
                              input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(conv6b)
    pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2, 2))(pr6)
    inter5 = concatenate([upconv5_c, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv5)
    pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2, 2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv4)
    pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2, 2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv3)
    pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2, 2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv2)
    pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2, 2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv1)
    # pr1b = Activation(K.softmax)(pr1)

    far_1 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_1 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)
    far_2 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr2b)
    close_2 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr2b)
    far_3 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr3b)
    close_3 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr3b)
    far_4 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr4b)
    close_4 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr4b)
    far_5 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr5b)
    close_5 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr5b)
    far_6 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr6b)
    close_6 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr6b)

    far_p1 = Multiply()([far_1, far1])
    close_p1 = Multiply()([close_1, close1])
    pre1 = Add()([far_p1, close_p1])

    far_p2 = Multiply()([far_2, far2])
    close_p2 = Multiply()([close_2, close2])
    pre2 = Add()([far_p2, close_p2])

    far_p3 = Multiply()([far_3, far3])
    close_p3 = Multiply()([close_3, close3])
    pre3 = Add()([far_p3, close_p3])

    far_p4 = Multiply()([far_4, far4])
    close_p4 = Multiply()([close_4, close4])
    pre4 = Add()([far_p4, close_p4])

    far_p5 = Multiply()([far_5, far5])
    close_p5 = Multiply()([close_5, close5])
    pre5 = Add()([far_p5, close_p5])

    far_p6 = Multiply()([far_6, far6])
    close_p6 = Multiply()([close_6, close6])
    pre6 = Add()([far_p6, close_p6])

    model = Model(inputs=[a, close6, close5, close4, close3, close2, close1, far6, far5, far4, far3, far2, far1],
                  outputs=(pre6, pre5, pre4, pre3, pre2, pre1))

    return model


def model_overall(model_c, model_f, model_j):
    model_input = Input(shape=(448, 640, 6))

    far = model_f(model_input)
    close = model_c(model_input)

    model_output = model_j([model_input,
                                    close[0], close[1], close[2], close[3], close[4], close[5],
                                    far[0], far[1], far[2], far[3], far[4], far[5]])

    model = Model(inputs=model_input, outputs=model_output)

    return model


def model_overall_shared(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    upconv5_c = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_c = UpSampling2D(size=(2,2))(pr6_c)
    inter5_c = concatenate([upconv5_c, conv5b, pr6up_c], axis=3)

    iconv5_c = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_c)
    pr5_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_c)
    pr5up_c = UpSampling2D(size=(2,2))(pr5_c)
    upconv4_c = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_c)
    inter4_c = concatenate([upconv4_c, conv4b, pr5up_c], axis=3)

    iconv4_c = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_c)
    pr4_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_c)
    pr4up_c = UpSampling2D(size=(2,2))(pr4_c)
    upconv3_c = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_c)
    inter3_c = concatenate([upconv3_c, conv3b, pr4up_c], axis=3)

    iconv3_c = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_c)
    pr3_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_c)
    pr3up_c = UpSampling2D(size=(2,2))(pr3_c)
    upconv2_c = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_c)
    inter2_c = concatenate([upconv2_c, conv2, pr3up_c], axis=3)

    iconv2_c = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_c)
    pr2_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_c)
    pr2up_c = UpSampling2D(size=(2,2))(pr2_c)
    upconv1_c = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_c)
    inter1_c = concatenate([upconv1_c, conv1, pr2up_c], axis=3)

    iconv1_c = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_c)
    pr1_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_c)

    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    upconv5_f  = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_f  = UpSampling2D(size=(2,2))(pr6_f )
    inter5_f  = concatenate([upconv5_f , conv5b, pr6up_f ], axis=3)

    iconv5_f  = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_f )
    pr5_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_f )
    pr5up_f  = UpSampling2D(size=(2,2))(pr5_f )
    upconv4_f  = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_f )
    inter4_f  = concatenate([upconv4_f , conv4b, pr5up_f ], axis=3)

    iconv4_f  = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_f )
    pr4_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_f )
    pr4up_f  = UpSampling2D(size=(2,2))(pr4_f )
    upconv3_f  = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_f )
    inter3_f  = concatenate([upconv3_f , conv3b, pr4up_f ], axis=3)

    iconv3_f  = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_f )
    pr3_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_f )
    pr3up_f  = UpSampling2D(size=(2,2))(pr3_f )
    upconv2_f  = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_f )
    inter2_f  = concatenate([upconv2_f , conv2, pr3up_f ], axis=3)

    iconv2_f  = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_f )
    pr2_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_f )
    pr2up_f  = UpSampling2D(size=(2,2))(pr2_f )
    upconv1_f  = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_f )
    inter1_f  = concatenate([upconv1_f , conv1, pr2up_f ], axis=3)

    iconv1_f  = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_f )
    pr1_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_f )

    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    upconv5_u  = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_u  = UpSampling2D(size=(2,2))(pr6_u )
    inter5_u  = concatenate([upconv5_u , conv5b, pr6up_u ], axis=3)

    iconv5_u  = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_u )
    pr5_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_u )
    pr5up_u  = UpSampling2D(size=(2,2))(pr5_u )
    upconv4_u  = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_u )
    inter4_u  = concatenate([upconv4_u , conv4b, pr5up_u ], axis=3)

    iconv4_u  = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_u )
    pr4_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_u )
    pr4up_u  = UpSampling2D(size=(2,2))(pr4_u )
    upconv3_u  = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_u )
    inter3_u  = concatenate([upconv3_u , conv3b, pr4up_u ], axis=3)

    iconv3_u  = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_u )
    pr3_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_u )
    pr3up_u  = UpSampling2D(size=(2,2))(pr3_u )
    upconv2_u  = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_u )
    inter2_u  = concatenate([upconv2_u , conv2, pr3up_u ], axis=3)

    iconv2_u  = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_u )
    pr2_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_u )
    pr2up_u  = UpSampling2D(size=(2,2))(pr2_u )
    upconv1_u  = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_u )
    inter1_u  = concatenate([upconv1_u , conv1, pr2up_u ], axis=3)

    iconv1_u  = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_u )
    pr1_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_u )

    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################

    upconv5 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5)
    pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4)
    pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3)
    pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2)
    pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv1)

    far_1 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_1 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)
    uncondrolled_1 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr1)
    far_2 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr2b)
    close_2 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr2b)
    uncondrolled_2 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr2b)
    far_3 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr3b)
    close_3 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr3b)
    uncondrolled_3 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr3b)
    far_4 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr4b)
    close_4 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr4b)
    uncondrolled_4 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr4b)
    far_5 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr5b)
    close_5 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr5b)
    uncondrolled_5 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr5b)
    far_6 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr6b)
    close_6 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr6b)
    uncondrolled_6 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr6b)

    far_p1 = Multiply()([far_1, pr1_f])
    close_p1 = Multiply()([close_1, pr1_c])
    uncondrolled_p1 = Multiply()([uncondrolled_1, pr1_u])
    pre1 = Add()([far_p1, close_p1, uncondrolled_p1])

    far_p2 = Multiply()([far_2, pr2_f])
    close_p2 = Multiply()([close_2, pr2_c])
    uncondrolled_p2 = Multiply()([uncondrolled_2, pr2_u])
    pre2 = Add()([far_p2, close_p2, uncondrolled_p2])

    far_p3 = Multiply()([far_3, pr3_f])
    close_p3 = Multiply()([close_3, pr3_c])
    uncondrolled_p3 = Multiply()([uncondrolled_3, pr3_u])
    pre3 = Add()([far_p3, close_p3, uncondrolled_p3])

    far_p4 = Multiply()([far_4, pr4_f])
    close_p4 = Multiply()([close_4, pr4_c])
    uncondrolled_p4 = Multiply()([uncondrolled_4, pr4_u])
    pre4 = Add()([far_p4, close_p4, uncondrolled_p4])

    far_p5 = Multiply()([far_5, pr5_f])
    close_p5 = Multiply()([close_5, pr5_c])
    uncondrolled_p5 = Multiply()([uncondrolled_5, pr5_u])
    pre5 = Add()([far_p5, close_p5, uncondrolled_p5])

    far_p6 = Multiply()([far_6, pr6_f])
    close_p6 = Multiply()([close_6, pr6_c])
    uncondrolled_p6 = Multiply()([uncondrolled_6, pr6_u])
    pre6 = Add()([far_p6, close_p6, uncondrolled_p6])


    model = Model(inputs=a, outputs=[pre6, pre5, pre4, pre3, pre2, pre1])

    return model

def model_overall_shared_show(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    upconv5_c = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_c = UpSampling2D(size=(2,2))(pr6_c)
    inter5_c = concatenate([upconv5_c, conv5b, pr6up_c], axis=3)

    iconv5_c = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_c)
    pr5_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_c)
    pr5up_c = UpSampling2D(size=(2,2))(pr5_c)
    upconv4_c = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_c)
    inter4_c = concatenate([upconv4_c, conv4b, pr5up_c], axis=3)

    iconv4_c = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_c)
    pr4_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_c)
    pr4up_c = UpSampling2D(size=(2,2))(pr4_c)
    upconv3_c = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_c)
    inter3_c = concatenate([upconv3_c, conv3b, pr4up_c], axis=3)

    iconv3_c = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_c)
    pr3_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_c)
    pr3up_c = UpSampling2D(size=(2,2))(pr3_c)
    upconv2_c = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_c)
    inter2_c = concatenate([upconv2_c, conv2, pr3up_c], axis=3)

    iconv2_c = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_c)
    pr2_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_c)
    pr2up_c = UpSampling2D(size=(2,2))(pr2_c)
    upconv1_c = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_c)
    inter1_c = concatenate([upconv1_c, conv1, pr2up_c], axis=3)

    iconv1_c = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_c)
    pr1_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_c)

    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    upconv5_f  = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_f  = UpSampling2D(size=(2,2))(pr6_f )
    inter5_f  = concatenate([upconv5_f , conv5b, pr6up_f ], axis=3)

    iconv5_f  = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_f )
    pr5_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_f )
    pr5up_f  = UpSampling2D(size=(2,2))(pr5_f )
    upconv4_f  = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_f )
    inter4_f  = concatenate([upconv4_f , conv4b, pr5up_f ], axis=3)

    iconv4_f  = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_f )
    pr4_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_f )
    pr4up_f  = UpSampling2D(size=(2,2))(pr4_f )
    upconv3_f  = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_f )
    inter3_f  = concatenate([upconv3_f , conv3b, pr4up_f ], axis=3)

    iconv3_f  = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_f )
    pr3_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_f )
    pr3up_f  = UpSampling2D(size=(2,2))(pr3_f )
    upconv2_f  = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_f )
    inter2_f  = concatenate([upconv2_f , conv2, pr3up_f ], axis=3)

    iconv2_f  = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_f )
    pr2_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_f )
    pr2up_f  = UpSampling2D(size=(2,2))(pr2_f )
    upconv1_f  = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_f )
    inter1_f  = concatenate([upconv1_f , conv1, pr2up_f ], axis=3)

    iconv1_f  = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_f )
    pr1_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_f )

    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    ############################## uncontrolled ###############################
    upconv5_u  = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_u  = UpSampling2D(size=(2,2))(pr6_u )
    inter5_u  = concatenate([upconv5_u , conv5b, pr6up_u ], axis=3)

    iconv5_u  = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_u )
    pr5_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_u )
    pr5up_u  = UpSampling2D(size=(2,2))(pr5_u )
    upconv4_u  = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_u )
    inter4_u  = concatenate([upconv4_u , conv4b, pr5up_u ], axis=3)

    iconv4_u  = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_u )
    pr4_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_u )
    pr4up_u  = UpSampling2D(size=(2,2))(pr4_u )
    upconv3_u  = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_u )
    inter3_u  = concatenate([upconv3_u , conv3b, pr4up_u ], axis=3)

    iconv3_u  = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_u )
    pr3_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_u )
    pr3up_u  = UpSampling2D(size=(2,2))(pr3_u )
    upconv2_u  = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_u )
    inter2_u  = concatenate([upconv2_u , conv2, pr3up_u ], axis=3)

    iconv2_u  = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_u )
    pr2_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_u )
    pr2up_u  = UpSampling2D(size=(2,2))(pr2_u )
    upconv1_u  = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_u )
    inter1_u  = concatenate([upconv1_u , conv1, pr2up_u ], axis=3)

    iconv1_u  = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_u )
    pr1_u  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_u )

    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################

    upconv5 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5)
    pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4)
    pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3)
    pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2)
    pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv1)

    far_1 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_1 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)
    uncondrolled_1 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr1)
    far_2 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr2b)
    close_2 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr2b)
    uncondrolled_2 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr2b)
    far_3 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr3b)
    close_3 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr3b)
    uncondrolled_3 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr3b)
    far_4 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr4b)
    close_4 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr4b)
    uncondrolled_4 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr4b)
    far_5 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr5b)
    close_5 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr5b)
    uncondrolled_5 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr5b)
    far_6 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr6b)
    close_6 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr6b)
    uncondrolled_6 = core.Lambda(lambda x: x[:, :, :, 2:3])(pr6b)

    far_p1 = Multiply()([far_1, pr1_f])
    close_p1 = Multiply()([close_1, pr1_c])
    uncondrolled_p1 = Multiply()([uncondrolled_1, pr1_u])
    pre1 = Add()([far_p1, close_p1, uncondrolled_p1])

    far_p2 = Multiply()([far_2, pr2_f])
    close_p2 = Multiply()([close_2, pr2_c])
    uncondrolled_p2 = Multiply()([uncondrolled_2, pr2_u])
    pre2 = Add()([far_p2, close_p2, uncondrolled_p2])

    far_p3 = Multiply()([far_3, pr3_f])
    close_p3 = Multiply()([close_3, pr3_c])
    uncondrolled_p3 = Multiply()([uncondrolled_3, pr3_u])
    pre3 = Add()([far_p3, close_p3, uncondrolled_p3])

    far_p4 = Multiply()([far_4, pr4_f])
    close_p4 = Multiply()([close_4, pr4_c])
    uncondrolled_p4 = Multiply()([uncondrolled_4, pr4_u])
    pre4 = Add()([far_p4, close_p4, uncondrolled_p4])

    far_p5 = Multiply()([far_5, pr5_f])
    close_p5 = Multiply()([close_5, pr5_c])
    uncondrolled_p5 = Multiply()([uncondrolled_5, pr5_u])
    pre5 = Add()([far_p5, close_p5, uncondrolled_p5])

    far_p6 = Multiply()([far_6, pr6_f])
    close_p6 = Multiply()([close_6, pr6_c])
    uncondrolled_p6 = Multiply()([uncondrolled_6, pr6_u])
    pre6 = Add()([far_p6, close_p6, uncondrolled_p6])


    model = Model(inputs=a, outputs=[pr6_c, pr5_c, pr4_c, pr3_c, pr2_c, pr1_c,
                                     pr6_f, pr5_f, pr4_f, pr3_f, pr2_f, pr1_f,
                                     pre6, pre5, pre4, pre3, pre2, pre1,
                                     close_6, close_5, close_4, close_3, close_2, close_1,
                                     pr6_u, pr5_u, pr4_u, pr3_u, pr2_u, pr1_u,
                                     uncondrolled_6, uncondrolled_5, uncondrolled_4, uncondrolled_3, uncondrolled_2, uncondrolled_1])

    return model


def model_overall_shared_old(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    upconv5_c = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_c = UpSampling2D(size=(2,2))(pr6_c)
    inter5_c = concatenate([upconv5_c, conv5b, pr6up_c], axis=3)

    iconv5_c = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_c)
    pr5_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_c)
    pr5up_c = UpSampling2D(size=(2,2))(pr5_c)
    upconv4_c = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_c)
    inter4_c = concatenate([upconv4_c, conv4b, pr5up_c], axis=3)

    iconv4_c = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_c)
    pr4_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_c)
    pr4up_c = UpSampling2D(size=(2,2))(pr4_c)
    upconv3_c = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_c)
    inter3_c = concatenate([upconv3_c, conv3b, pr4up_c], axis=3)

    iconv3_c = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_c)
    pr3_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_c)
    pr3up_c = UpSampling2D(size=(2,2))(pr3_c)
    upconv2_c = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_c)
    inter2_c = concatenate([upconv2_c, conv2, pr3up_c], axis=3)

    iconv2_c = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_c)
    pr2_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_c)
    pr2up_c = UpSampling2D(size=(2,2))(pr2_c)
    upconv1_c = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_c)
    inter1_c = concatenate([upconv1_c, conv1, pr2up_c], axis=3)

    iconv1_c = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_c)
    pr1_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_c)

    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    upconv5_f  = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_f  = UpSampling2D(size=(2,2))(pr6_f )
    inter5_f  = concatenate([upconv5_f , conv5b, pr6up_f ], axis=3)

    iconv5_f  = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_f )
    pr5_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_f )
    pr5up_f  = UpSampling2D(size=(2,2))(pr5_f )
    upconv4_f  = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_f )
    inter4_f  = concatenate([upconv4_f , conv4b, pr5up_f ], axis=3)

    iconv4_f  = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_f )
    pr4_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_f )
    pr4up_f  = UpSampling2D(size=(2,2))(pr4_f )
    upconv3_f  = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_f )
    inter3_f  = concatenate([upconv3_f , conv3b, pr4up_f ], axis=3)

    iconv3_f  = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_f )
    pr3_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_f )
    pr3up_f  = UpSampling2D(size=(2,2))(pr3_f )
    upconv2_f  = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_f )
    inter2_f  = concatenate([upconv2_f , conv2, pr3up_f ], axis=3)

    iconv2_f  = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_f )
    pr2_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_f )
    pr2up_f  = UpSampling2D(size=(2,2))(pr2_f )
    upconv1_f  = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_f )
    inter1_f  = concatenate([upconv1_f , conv1, pr2up_f ], axis=3)

    iconv1_f  = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_f )
    pr1_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_f )

    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################

    upconv5 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5)
    pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4)
    pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3)
    pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2)
    pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv1)

    far_1 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_1 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)
    far_2 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr2b)
    close_2 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr2b)
    far_3 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr3b)
    close_3 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr3b)
    far_4 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr4b)
    close_4 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr4b)
    far_5 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr5b)
    close_5 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr5b)
    far_6 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr6b)
    close_6 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr6b)

    far_p1 = Multiply()([far_1, pr1_f])
    close_p1 = Multiply()([close_1, pr1_c])
    pre1 = Add()([far_p1, close_p1])

    far_p2 = Multiply()([far_2, pr2_f])
    close_p2 = Multiply()([close_2, pr2_c])
    pre2 = Add()([far_p2, close_p2])

    far_p3 = Multiply()([far_3, pr3_f])
    close_p3 = Multiply()([close_3, pr3_c])
    pre3 = Add()([far_p3, close_p3])

    far_p4 = Multiply()([far_4, pr4_f])
    close_p4 = Multiply()([close_4, pr4_c])
    pre4 = Add()([far_p4, close_p4])

    far_p5 = Multiply()([far_5, pr5_f])
    close_p5 = Multiply()([close_5, pr5_c])
    pre5 = Add()([far_p5, close_p5])

    far_p6 = Multiply()([far_6, pr6_f])
    close_p6 = Multiply()([close_6, pr6_c])
    pre6 = Add()([far_p6, close_p6])


    model = Model(inputs=a, outputs=[pr6_c, pr5_c, pr4_c, pr3_c, pr2_c, pr1_c,
                                     pr6_f, pr5_f, pr4_f, pr3_f, pr2_f, pr1_f,
                                     pre6, pre5, pre4, pre3, pre2, pre1])

    return model

def model_overall_shared_old_show(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    ############################## close ###############################
    upconv5_c = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_c = UpSampling2D(size=(2,2))(pr6_c)
    inter5_c = concatenate([upconv5_c, conv5b, pr6up_c], axis=3)

    iconv5_c = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_c)
    pr5_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_c)
    pr5up_c = UpSampling2D(size=(2,2))(pr5_c)
    upconv4_c = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_c)
    inter4_c = concatenate([upconv4_c, conv4b, pr5up_c], axis=3)

    iconv4_c = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_c)
    pr4_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_c)
    pr4up_c = UpSampling2D(size=(2,2))(pr4_c)
    upconv3_c = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_c)
    inter3_c = concatenate([upconv3_c, conv3b, pr4up_c], axis=3)

    iconv3_c = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_c)
    pr3_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_c)
    pr3up_c = UpSampling2D(size=(2,2))(pr3_c)
    upconv2_c = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_c)
    inter2_c = concatenate([upconv2_c, conv2, pr3up_c], axis=3)

    iconv2_c = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_c)
    pr2_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_c)
    pr2up_c = UpSampling2D(size=(2,2))(pr2_c)
    upconv1_c = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_c)
    inter1_c = concatenate([upconv1_c, conv1, pr2up_c], axis=3)

    iconv1_c = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_c)
    pr1_c = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_c)

    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    ############################## far ###############################
    upconv5_f  = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up_f  = UpSampling2D(size=(2,2))(pr6_f )
    inter5_f  = concatenate([upconv5_f , conv5b, pr6up_f ], axis=3)

    iconv5_f  = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5_f )
    pr5_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5_f )
    pr5up_f  = UpSampling2D(size=(2,2))(pr5_f )
    upconv4_f  = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5_f )
    inter4_f  = concatenate([upconv4_f , conv4b, pr5up_f ], axis=3)

    iconv4_f  = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4_f )
    pr4_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4_f )
    pr4up_f  = UpSampling2D(size=(2,2))(pr4_f )
    upconv3_f  = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4_f )
    inter3_f  = concatenate([upconv3_f , conv3b, pr4up_f ], axis=3)

    iconv3_f  = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3_f )
    pr3_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3_f )
    pr3up_f  = UpSampling2D(size=(2,2))(pr3_f )
    upconv2_f  = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3_f )
    inter2_f  = concatenate([upconv2_f , conv2, pr3up_f ], axis=3)

    iconv2_f  = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2_f )
    pr2_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2_f )
    pr2up_f  = UpSampling2D(size=(2,2))(pr2_f )
    upconv1_f  = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2_f )
    inter1_f  = concatenate([upconv1_f , conv1, pr2up_f ], axis=3)

    iconv1_f  = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1_f )
    pr1_f  = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1_f )

    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################
    ############################## fuse ###############################

    upconv5 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5)
    pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4)
    pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3)
    pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2)
    pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=16, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv1)

    far_1 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_1 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)
    far_2 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr2b)
    close_2 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr2b)
    far_3 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr3b)
    close_3 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr3b)
    far_4 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr4b)
    close_4 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr4b)
    far_5 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr5b)
    close_5 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr5b)
    far_6 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr6b)
    close_6 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr6b)

    far_p1 = Multiply()([far_1, pr1_f])
    close_p1 = Multiply()([close_1, pr1_c])
    pre1 = Add()([far_p1, close_p1])

    far_p2 = Multiply()([far_2, pr2_f])
    close_p2 = Multiply()([close_2, pr2_c])
    pre2 = Add()([far_p2, close_p2])

    far_p3 = Multiply()([far_3, pr3_f])
    close_p3 = Multiply()([close_3, pr3_c])
    pre3 = Add()([far_p3, close_p3])

    far_p4 = Multiply()([far_4, pr4_f])
    close_p4 = Multiply()([close_4, pr4_c])
    pre4 = Add()([far_p4, close_p4])

    far_p5 = Multiply()([far_5, pr5_f])
    close_p5 = Multiply()([close_5, pr5_c])
    pre5 = Add()([far_p5, close_p5])

    far_p6 = Multiply()([far_6, pr6_f])
    close_p6 = Multiply()([close_6, pr6_c])
    pre6 = Add()([far_p6, close_p6])


    model = Model(inputs=a, outputs=[pr6_c, pr5_c, pr4_c, pr3_c, pr2_c, pr1_c,
                                     pr6_f, pr5_f, pr4_f, pr3_f, pr2_f, pr1_f,
                                     pre6, pre5, pre4, pre3, pre2, pre1,
                                     close_6, close_5, close_4, close_3, close_2, close_1])

    return model
