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

batchSize = 10
save_path = '/media/mjia/Data/SUN3D/filled/'
path = '/media/mjia/Data/SUN3D/train/'

# input image dimensions
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 4)

# initialize the model
model = model_ini.model_fill_hole(input_shape)

model.compile(loss=utility.my_loss,
              metrics=[utility.metric_L1_real],
              optimizer=keras.optimizers.Adadelta())

model.load_weights('../../exp_data/trained_models/model_epoch_39.hdf5')
index_org = sio.loadmat('Shuffled_index.mat')

# preper data
image_mean = np.zeros(shape=(448, 640, 3))
image_mean[:, :, 0] = 114 * np.ones(shape=(448, 640))
image_mean[:, :, 1] = 105 * np.ones(shape=(448, 640))
image_mean[:, :, 2] = 97 * np.ones(shape=(448, 640))

index_all = sio.loadmat('train_val_test_index.mat');
index = index_all['total'].transpose()
i = 0
while (True):
    (x, y, name) = utility.loadData_fill_hole_run(index, i, batchSize, path, image_mean)
    yp = model.predict_on_batch(x)
    i = i + 1

    for j in range(10):
        this = yp[5][j, :, :, 0]
        save_name =save_path + str(int(name[j, 0])).zfill(7) + '.mat'
        sio.savemat(save_name, {'filled_depth': this})