from __future__ import print_function

from random import shuffle

import cv2
import keras
import numpy as np
import scipy.io as sio
import tensorflow as tf
import utility
from keras import backend as K
from keras import metrics

import model_ini

trainn = 14409
val = 750

def my_loss_hole(y_true, y_pred):
    return K.mean(tf.div(tf.multiply(K.square(y_pred - y_true), y_true), tf.add(y_true, 1e-9*tf.ones_like(y_true))), axis=-1)

def my_loss(y_true, y_pred):
    return K.mean(tf.div(tf.multiply(K.square(y_pred - y_true), y_true), tf.add(y_true, 1e-9*tf.ones_like(y_true))), axis=-1)
    #return K.mean(tf.multiply(K.square(y_pred - y_true), tf.div(y_true, y_true)), axis=-1)

def metric_L1_real(y_true, y_pred):
    return K.mean(tf.div(tf.multiply(K.abs(y_pred-y_true), y_true), tf.add(K.square(y_true), 1e-9*tf.ones_like(y_true))))

def metric_L1_inv(y_true, y_pred):
    return K.mean(K.abs(tf.subtract(tf.div(tf.ones_like(y_pred), y_pred), tf.div(tf.multiply(tf.ones_like(y_true), y_true), tf.add(K.square(y_true), 1e-5*tf.ones_like(y_true))))))

def loadData(index, index_begin, batchSize, path):
    x = np.empty(shape=(batchSize, 448, 640, 6))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i][0])
        filename = path + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:]
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:]
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = cv2.pyrDown(yy)
        y2[i, :, :, 0] = cv2.pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = cv2.pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = cv2.pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = cv2.pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = cv2.pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1]

    return (x,y)

def data_generator(isTrain = True, batchSize = 10):
    if isTrain:
        path = '/media/mjia/Data/SUN3D/train/'
        index = [[i] for i in range(1,trainn)]
        shuffle(index)
    else:
        index = [[i] for i in range(1,val)]
        shuffle(index)
        path = '/media/mjia/Data/SUN3D/val/'

    i = 0
    while(True):
        yield loadData(index, i, batchSize, path)
        i = i + batchSize





# input image dimensions
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 4)

# initialize the model
model = model_ini.model_fill_hole(input_shape)

model.compile(loss=utility.my_loss,
              metrics=[utility.metric_L1_real],
              optimizer=keras.optimizers.Adadelta())

#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam())

index_org = sio.loadmat('Shuffled_index.mat')
model.load_weights('../exp_data/trained_models/model_epoch_3.hdf5')

#for i in range(20):
 #   model = train.train_epoch(model)
  #  filename = './trained_models/model_epoch_' + str(i) + '.hdf5'
   # model.save_weights(filename)
#plot_model(model, to_file='./trained_models/model.png')
loss = np.empty(shape=(40, 13))

for i in range(4, 40):

    history = model.fit_generator(utility.data_generator(index_org['index'], isTrain = True, isGAN = False, close_far_all = 5, batchSize = 10), steps_per_epoch = 6000, epochs = 1)
    loss[i] = model.evaluate_generator(utility.data_generator(index_org['index'], isTrain = False, isGAN = False, close_far_all = 5, batchSize = 20), steps = 400)
    filename = '../exp_data/trained_models/model_epoch_' + str(i) + '.hdf5'
    model.save_weights(filename)
    filename = '../exp_data/trained_models/model_epoch_train' + str(i)
    np.save(filename, history.history)
    filename = '../exp_data/trained_models/model_epoch_val' + str(i)
    np.save(filename, loss[i])

print('\n')
np.save('../exp_data/trained_models/loss', loss)