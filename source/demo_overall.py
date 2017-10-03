from __future__ import print_function

from random import shuffle
import keras
import utility
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from PIL import Image
from keras import backend as K
import cv2

import model_ini

trainn = 26000
val = 8000

total = 1147
path = '/media/mjia/Data/SUN3D/demo2/'

def zero_mask(y):
    return tf.to_float(K.not_equal(K.zeros_like(y), y))

def zero_mask_inv(y):
    return tf.to_float(K.equal(K.zeros_like(y), y))

def my_loss(y_true, y_pred):
    return K.mean(tf.multiply(K.square(y_true - y_pred), zero_mask(y_true)))
    #return K.mean(tf.multiply(K.square(y_pred - y_true), tf.div(y_true, y_true)), axis=-1)

def metric_L1_real(y_true, y_pred):
    return K.mean(tf.realdiv(tf.multiply(K.abs(y_pred-y_true), zero_mask(y_true)), tf.add(y_true, zero_mask_inv(y_true))))

def metric_L1_inv(y_true, y_pred):
    return K.mean(K.abs(tf.realdiv(zero_mask(y_true), y_pred) - tf.realdiv(zero_mask(y_true), tf.add(y_true, zero_mask_inv(y_true)))))

def showImange(x, yy, depth, batchSize):
    img = np.empty(shape=(batchSize, 224, 960, 3))
    xx = np.empty(shape=(batchSize, 224, 320, 6))
    xx[:, :, :, :] = x[:, ::2, ::2, :]
    for i in range(batchSize):
        #imgx = Image.fromarray(x[i][:,:,0:3], 'RGB')
        #imgx.show()
        #imgy = Image.fromarray(30*np.float32(y[5][i][:,:,0]), 'F')
        #imgy.show()
        #imgd = Image.fromarray(30*depth[5][i][:, :, 0], 'F')
        #imgd.show()
        img[i][:,0:320,:] = np.float32(xx[i][:,:,3:6])
        img[i][:,320:640,0] = 30*np.float32(yy)
        img[i][:,320:640,1] = 30*np.float32(yy)
        img[i][:,320:640,2] = 30*np.float32(yy)
        img[i][:,640:960,0] = 30*depth[i][:,:,0]
        img[i][:,640:960,1] = 30*depth[i][:,:,0]
        img[i][:,640:960,2] = 30*depth[i][:,:,0]
        imgtoshow = Image.fromarray(np.uint8(img[i]), 'RGB')
        return imgtoshow

def save_result_old():
    path = '/media/mjia/Data/SUN3D/train/'
    index_org = sio.loadmat('train_val_test_index.mat')
    index = index_org['test_index'].transpose()
    [x, y] = utility.loadData_fill_hole(index, 70, 15, path, image_mean)

    depth = model_old.predict_on_batch(x)
    plus = cv2.imread('./draw/plus.jpg')
    mult = cv2.imread('./draw/multiply.jpg')
    equal = cv2.imread('./draw/equal.jpg')
    empty = cv2.imread('./draw/empty.jpg')

    sep = np.ones(shape=(50, 420*4-100, 3))
    img = np.ones(shape=(4110, 420*4-100, 3))
    for i in range(15):
        image_to_save = np.zeros(shape=(224, 420*4-100, 3))

        image_to_save[:, 0:320, :] = 255*x[i, ::2, ::2, 0:3] + image_mean[::2, ::2, :]

        image_to_save[:, 320:420, :] = empty

        image_to_save[:, 420:740, 0] = 255 * x[i, ::2, ::2, 3]
        image_to_save[:, 420:740, 1] = 255 * x[i, ::2, ::2, 3]
        image_to_save[:, 420:740, 2] = 255 * x[i, ::2, ::2, 3]

        image_to_save[:, 740:840, :] = empty

        image_to_save[:, 840:1160, 0] = 250 * y[5][i, :, :, 0]
        image_to_save[:, 840:1160, 1] = 250 * y[5][i, :, :, 0]
        image_to_save[:, 840:1160, 2] = 250 * y[5][i, :, :, 0]

        image_to_save[:, 1160:1260, :] = empty

        image_to_save[:, 1260:1580, 0] = 250 * depth[5][i, :, :, 0]
        image_to_save[:, 1260:1580, 1] = 250 * depth[5][i, :, :, 0]
        image_to_save[:, 1260:1580, 2] = 250 * depth[5][i, :, :, 0]

        img[(274*i):(274*i+224), :, :] = image_to_save
        img[(274*i+224):(274*i+274), :, :] = 255*sep

    cv2.imwrite('../demo_images/epoch_33_test.jpg', img)



# initialize the model
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 4)
#model_close = model_ini.model_init(input_shape)
#model_far = model_ini.model_init(input_shape)
#model_judge = model_ini.model_judgement2()
model_old = model_ini.model_fill_hole(input_shape)

model_old.load_weights('../../exp_data/trained_models/model_epoch_33.hdf5')

#loss = model.evaluate_generator(utility.data_generator(isTrain = False, isGAN= False, batchSize = 20), steps = 255)

x = np.empty(shape=(1, 448, 640, 6))
image_mean = np.zeros(shape=(448, 640, 3))
image_mean[:, :, 0] = 114 * np.ones(shape=(448, 640))
image_mean[:, :, 1] = 105 * np.ones(shape=(448, 640))
image_mean[:, :, 2] = 97 * np.ones(shape=(448, 640))
#image_mean[:, :, 3] = 114 * np.ones(shape=(448, 640))
#image_mean[:, :, 4] = 105 * np.ones(shape=(448, 640))
#image_mean[:, :, 5] = 97 * np.ones(shape=(448, 640))
#my_video = cv2.VideoWriter(filename='video.avi', fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), fps=10, frameSize=(224, 960), isColor=True)

#for i in range(1, 20):
#    x = np.empty(shape=(1, 448, 640, 6))
#    filename = '/media/mjia/Data/SUN3D/val/' + str(np.random.randint(1, 750)).zfill(7) + '.mat'
#    xx = sio.loadmat(filename)
#    x[:, :, :, 0:3] = xx['Data']['image'][0][0][0][0][16:464, :, :] - image_mean
#    x /= 255
#    depth = model.predict_on_batch(x)

#    img = xx['Data']['image'][0][0][0][1][16:464, :, :]
#    depth_gt = xx['Data']['depth'][0][0][0][1][16:464, :]
#    dict_to_save = {'img': img, 'depth': depth[5], 'depth_gt': depth_gt}
#    filename = './For_Yiming/model1_val/' + str(i).zfill(3) + '.mat'
#    sio.savemat(filename, dict_to_save)

save_result_old()

while True:
    path = '/media/mjia/Data/SUN3D/train/'
    index = [[i] for i in range(1, 102899)]
    shuffle(index)
    ([x, x1, x2], y1) = utility.loadData_judgement(index, 50, 10, path, image_mean)
    depth = model_overall.predict_on_batch(x)

    image_to_show = np.zeros(shape=(10, 672, 640))
    image_to_show[:,0:224,0:320] = depth[5][:,:,:,0]
    image_to_show[:,0:224,320:640] = y1[:,:,:,0]
    image_to_show[:,224:448,0:320] = depth[11][:,:,:,0]
    image_to_show[:,224:448,320:640] = depth[17][:,:,:,0]
    image_to_show[:,448:672,0:320] = 5 * depth[23][:,:,:,0]
    # image_to_show.close()
    # image_to_show.show(title=1)
    for i in range(10):
        plt.imshow(image_to_show[i])