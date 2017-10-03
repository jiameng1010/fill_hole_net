from __future__ import print_function

from numpy.random import shuffle

import cv2
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

trainn = 26000
val = 8000
train_path = '/media/mjia/Data/SUN3D/train/'
val_path = '/media/mjia/Data/SUN3D/val/'
data_path = '/media/mjia/Data/SUN3D/train/'


def loadDataGAN(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 6))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    label = np.ones(shape=(batchSize, 2))
    label[:, 1] = np.zeros_like(label[:, 1])
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i][0])
        filename = path + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1, label]

    return (x,y)

def loadData_judgement(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 6))
    x1 = np.ones(shape=(batchSize, 224, 320, 1))
    x2 = np.ones(shape=(batchSize, 224, 320, 1))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i][0])
        filename = path + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = pyrDown(yy)

        #filename = path + number_of_file.zfill(7) + '.npy'
        #xx = np.load(filename)
        #x1[i,:,:,:] = xx[0]
        #x2[i,:,:,:] = xx[1]

    ind_zero = y1[:,:,:,:] < 0.3
    y1[ind_zero] = 0
    x = x.astype('float32')
    x /= 255

    return ([x, x1, x2], y1)

def loadData_close(index, index_begin, batchSize, path, image_mean):
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
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        ind = yy[:,:] > 3
        yy[ind] = 0
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1]

    return (x,y)


def loadData(index, index_begin, batchSize, path, image_mean):
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
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1]

    return (x,y)

def loadData_far(index, index_begin, batchSize, path, image_mean):
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
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        ind = yy[:,:] < 2.5
        yy[ind] = 0
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1]

    return (x,y)

def loadData_binary(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 6))
    yy = np.empty(shape=(448, 640))
    yy_NA = np.zeros(shape=(448, 640), dtype='float32')
    yy_far = np.zeros(shape=(448, 640), dtype='float32')
    y1 = np.empty(shape=(batchSize, 224, 320, 3))
    y2 = np.empty(shape=(batchSize, 112, 160, 3))
    y3 = np.empty(shape=(batchSize, 56, 80, 3))
    y4 = np.empty(shape=(batchSize, 28, 40, 3))
    y5 = np.empty(shape=(batchSize, 14, 20, 3))
    y6 = np.empty(shape=(batchSize, 7, 10, 3))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i][0])
        filename = path + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][0][16:464,:]
        yy = yy.astype('float32')
        ind_NA = yy[:,:] == 0
        ind_far = yy[:,:] > 4
        yy_NA[ind_NA] = 1
        yy_far[ind_far] = 1

        #NA
        y1[i, :, :, 0] = cv2.threshold(pyrDown(yy_NA), 0.5, 1, cv2.THRESH_BINARY)[1]
        y2[i, :, :, 0] = cv2.threshold(pyrDown(y1[i, :, :, 0]), 0.55, 1, cv2.THRESH_BINARY)[1]
        y3[i, :, :, 0] = cv2.threshold(pyrDown(y2[i, :, :, 0]), 0.55, 1, cv2.THRESH_BINARY)[1]
        y4[i, :, :, 0] = cv2.threshold(pyrDown(y3[i, :, :, 0]), 0.55, 1, cv2.THRESH_BINARY)[1]
        y5[i, :, :, 0] = cv2.threshold(pyrDown(y4[i, :, :, 0]), 0.55, 1, cv2.THRESH_BINARY)[1]
        y6[i, :, :, 0] = cv2.threshold(pyrDown(y5[i, :, :, 0]), 0.55, 1, cv2.THRESH_BINARY)[1]

        #far
        y1[i, :, :, 1] = cv2.threshold(pyrDown(yy_far), 0.5, 1, cv2.THRESH_BINARY)[1]
        y2[i, :, :, 1] = cv2.threshold(pyrDown(y1[i, :, :, 1]), 0.5, 1, cv2.THRESH_BINARY)[1]
        y3[i, :, :, 1] = cv2.threshold(pyrDown(y2[i, :, :, 1]), 0.5, 1, cv2.THRESH_BINARY)[1]
        y4[i, :, :, 1] = cv2.threshold(pyrDown(y3[i, :, :, 1]), 0.5, 1, cv2.THRESH_BINARY)[1]
        y5[i, :, :, 1] = cv2.threshold(pyrDown(y4[i, :, :, 1]), 0.5, 1, cv2.THRESH_BINARY)[1]
        y6[i, :, :, 1] = cv2.threshold(pyrDown(y5[i, :, :, 1]), 0.5, 1, cv2.THRESH_BINARY)[1]

        #close
        y1[i, :, :, 2] = np.ones_like(y1[i, :, :, 1]) - y1[i, :, :, 1]
        y2[i, :, :, 2] = np.ones_like(y2[i, :, :, 1]) - y2[i, :, :, 1]
        y3[i, :, :, 2] = np.ones_like(y3[i, :, :, 1]) - y3[i, :, :, 1]
        y4[i, :, :, 2] = np.ones_like(y4[i, :, :, 1]) - y4[i, :, :, 1]
        y5[i, :, :, 2] = np.ones_like(y5[i, :, :, 1]) - y5[i, :, :, 1]
        y6[i, :, :, 2] = np.ones_like(y6[i, :, :, 1]) - y6[i, :, :, 1]

    ind_na = y1[:,:,:,0] == 1
    y1[ind_na, 1] = 0
    y1[ind_na, 2] = 0
    ind_na = y2[:,:,:,0] == 1
    y2[ind_na, 1] = 0
    y2[ind_na, 2] = 0
    ind_na = y3[:,:,:,0] == 1
    y3[ind_na, 1] = 0
    y3[ind_na, 2] = 0
    ind_na = y4[:,:,:,0] == 1
    y4[ind_na, 1] = 0
    y4[ind_na, 2] = 0
    ind_na = y5[:,:,:,0] == 1
    y5[ind_na, 1] = 0
    y5[ind_na, 2] = 0
    ind_na = y6[:,:,:,0] == 1
    y6[ind_na, 1] = 0
    y6[ind_na, 2] = 0

    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1]

    return (x,y)


def train_GAN_epoch(model_d, generator, isTrain, batchSize = 10, steps=10):
    record = [0, 0]
    image_mean = np.zeros(shape=(448, 640, 3))
    image_mean[:,:,0] = 114*np.ones(shape=(448, 640))
    image_mean[:,:,1] = 105*np.ones(shape=(448, 640))
    image_mean[:,:,2] = 97*np.ones(shape=(448, 640))
    if isTrain:
        path = train_path
        index = [[i] for i in range(1,trainn)]
        shuffle(index)
    else:
        index = [[i] for i in range(1,val)]
        shuffle(index)
        path = val_path

    i = 0
    gt = np.empty(shape=(batchSize, 2))
    gt[0:int(batchSize/2), 0] = np.ones_like(gt[0:int(batchSize/2), 0])
    gt[0:int(batchSize/2), 1] = np.zeros_like(gt[0:int(batchSize/2), 1])
    gt[int(batchSize/2):batchSize, 0] = np.zeros_like(gt[int(batchSize/2):batchSize, 0])
    gt[int(batchSize/2):batchSize, 1] = np.ones_like(gt[int(batchSize/2):batchSize, 1])
    for step in range(steps):
        [x, y] = loadData(index, i, int(batchSize/2), path, image_mean)
        depth = generator.predict_on_batch(x)
        [x, y] = loadData(index, i+int(batchSize/2), int(batchSize/2), path, image_mean)
        y6 = np.concatenate((y[0], depth[0]), 0)
        y5 = np.concatenate((y[1], depth[1]), 0)
        y4 = np.concatenate((y[2], depth[2]), 0)
        y3 = np.concatenate((y[3], depth[3]), 0)
        y2 = np.concatenate((y[4], depth[4]), 0)
        y1 = np.concatenate((y[5], depth[5]), 0)
        if isTrain:
            loss = model_d.train_on_batch([y6, y5, y4, y3, y2, y1], gt)
            record = np.add(record, loss)
        else:
            loss = model_d.test_on_batch([y6, y5, y4, y3, y2, y1], gt)
            record = np.add(record, loss)
        i = i + batchSize
    if isTrain:
        return model_d
    else:
        record[0] = record[0] / steps
        record[1] = record[1] / steps
        return record

def fake_generator(isTrain, depth_generator, batchSize = 10):
    image_mean = np.zeros(shape=(448, 640, 3))
    image_mean[:,:,0] = 114*np.ones(shape=(448, 640))
    image_mean[:,:,1] = 105*np.ones(shape=(448, 640))
    image_mean[:,:,2] = 97*np.ones(shape=(448, 640))
    if isTrain:
        path = train_path
        index = [[i] for i in range(1,trainn)]
        shuffle(index)
    else:
        index = [[i] for i in range(1,val)]
        shuffle(index)
        path = val_path

    i = 0
    gt = np.empty(shape=(batchSize, 1))
    gt[0:int(batchSize/2), 0] = np.ones_like(gt[0:int(batchSize/2), 0])
    gt[int(batchSize/2):batchSize, 0] = np.zeros_like(gt[int(batchSize/2):batchSize, 0])
    while(True):
        [x, y] = loadData(index, i, int(batchSize/2), path, image_mean)
        depth = depth_generator.predict_on_batch(x)
        [x, y] = loadData(index, i+int(batchSize/2), int(batchSize/2), path, image_mean)
        y6 = np.concatenate((y[0], depth[0]), 0)
        y5 = np.concatenate((y[1], depth[1]), 0)
        y4 = np.concatenate((y[2], depth[2]), 0)
        y3 = np.concatenate((y[3], depth[3]), 0)
        y2 = np.concatenate((y[4], depth[4]), 0)
        y1 = np.concatenate((y[5], depth[5]), 0)
        yield ([y6, y5, y4, y3, y2, y1], gt)
        i = i + batchSize


def judgement_generator(isTrain = True, batchSize = 10):
    image_mean = np.zeros(shape=(448, 640, 3))
    image_mean[:,:,0] = 114*np.ones(shape=(448, 640))
    image_mean[:,:,1] = 105*np.ones(shape=(448, 640))
    image_mean[:,:,2] = 97*np.ones(shape=(448, 640))
    if isTrain:
        path = train_path
        index = [[i] for i in range(1,trainn)]
        shuffle(index)
    else:
        index = [[i] for i in range(1,val)]
        shuffle(index)
        path = val_path

    i = 0
    while(True):
        yield loadData_judgement(index, i, batchSize, path, image_mean)


def overall_generator(isTrain = True, batchSize = 10):
    image_mean = np.zeros(shape=(448, 640, 3))
    image_mean[:,:,0] = 114*np.ones(shape=(448, 640))
    image_mean[:,:,1] = 105*np.ones(shape=(448, 640))
    image_mean[:,:,2] = 97*np.ones(shape=(448, 640))
    if isTrain:
        path = train_path
        index = [[i] for i in range(1,trainn)]
        shuffle(index)
    else:
        index = [[i] for i in range(1,val)]
        shuffle(index)
        path = val_path

    i = 0
    while(True):
        yield loadData_overall(index, i, batchSize, path, image_mean)


def loadData_overall(index, index_begin, batchSize, path, image_mean):
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
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255

    y = [y6, y5, y4, y3, y2, y1]

    return (x, y)


def loadData_edge(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 6))
    yy = np.empty(shape=(448, 640))
    edge = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    y1e = np.empty(shape=(batchSize, 224, 320, 1))
    y2e = np.empty(shape=(batchSize, 112, 160, 1))
    y3e = np.empty(shape=(batchSize, 56, 80, 1))
    y4e = np.empty(shape=(batchSize, 28, 40, 1))
    y5e = np.empty(shape=(batchSize, 14, 20, 1))
    y6e = np.empty(shape=(batchSize, 7, 10, 1))
    y1ne = np.empty(shape=(batchSize, 224, 320, 1))
    y2ne = np.empty(shape=(batchSize, 112, 160, 1))
    y3ne = np.empty(shape=(batchSize, 56, 80, 1))
    y4ne = np.empty(shape=(batchSize, 28, 40, 1))
    y5ne = np.empty(shape=(batchSize, 14, 20, 1))
    y6ne = np.empty(shape=(batchSize, 7, 10, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i][0])
        filename = path + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        yy = xx['Data']['depth'][0][0][0][0][16:464,:]
        yy = yy.astype('float32')
        edge = xx['Data']['mask'][0][0][16:464,:]
        nonedge = np.ones_like(edge) - edge
        yye = np.multiply(yy, edge)
        yyne = np.multiply(yy, nonedge)
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])
        y1e[i, :, :, 0] = pyrDown(yye)
        y2e[i, :, :, 0] = pyrDown(y1e[i, :, :, 0])
        y3e[i, :, :, 0] = pyrDown(y2e[i, :, :, 0])
        y4e[i, :, :, 0] = pyrDown(y3e[i, :, :, 0])
        y5e[i, :, :, 0] = pyrDown(y4e[i, :, :, 0])
        y6e[i, :, :, 0] = pyrDown(y5e[i, :, :, 0])
        y1ne[i, :, :, 0] = pyrDown(yyne)
        y2ne[i, :, :, 0] = pyrDown(y1ne[i, :, :, 0])
        y3ne[i, :, :, 0] = pyrDown(y2ne[i, :, :, 0])
        y4ne[i, :, :, 0] = pyrDown(y3ne[i, :, :, 0])
        y5ne[i, :, :, 0] = pyrDown(y4ne[i, :, :, 0])
        y6ne[i, :, :, 0] = pyrDown(y5ne[i, :, :, 0])

    x = x.astype('float32')
    x /= 255
    y = [y6e, y5e, y4e, y3e, y2e, y1e, y6ne, y5ne, y4ne, y3ne, y2ne, y1ne, y6, y5, y4, y3, y2, y1]

    return (x,y)

def loadData_fill_hole_run(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 4))
    name = np.empty(shape=(batchSize, 1))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i, 0])
        filename = path + number_of_file.zfill(7) + '.mat'
        print(filename)
        name[i, 0] = index[index_begin+i, 0]
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        #x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        x[i,:,:,3] = xx['Data']['filled_depth'][0][0][16:464,:]*51
        yy = xx['Data']['filled_depth'][0][0][16:464,:]# - xx['Data']['depth'][0][0][0][0][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255

    y = [y6/5, y5/5, y4/5, y3/5, y2/5, y1/5]

    return (x, y, name)

def loadData_fill_hole(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 4))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i, 0])
        filename = path + number_of_file.zfill(7) + '.mat'
        print(filename)
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        #x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        x[i,:,:,3] = xx['Data']['depth'][0][0][0][1][16:464,:]*51
        yy = xx['Data']['filled_depth'][0][0][16:464,:]# - xx['Data']['depth'][0][0][0][0][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = pyrDown(yy)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255

    y = [y6/5, y5/5, y4/5, y3/5, y2/5, y1/5]

    return (x, y)

def loadData_predict_hole(index, index_begin, batchSize, path, image_mean):
    x = np.empty(shape=(batchSize, 448, 640, 3))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i])
        filename = path + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:] - image_mean      #for evaluate the monocular
        #x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:] - image_mean
        #x[i,:,:,3] = xx['Data']['depth'][0][0][0][1][16:464,:]*17
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        ind0 = yy[:, :] == 0
        mask = np.ones_like(yy)
        mask[ind0] = 0
        mask = mask.astype('float32')
        y1[i, :, :, 0] = pyrDown(mask)
        y2[i, :, :, 0] = pyrDown(y1[i, :, :, 0])
        y3[i, :, :, 0] = pyrDown(y2[i, :, :, 0])
        y4[i, :, :, 0] = pyrDown(y3[i, :, :, 0])
        y5[i, :, :, 0] = pyrDown(y4[i, :, :, 0])
        y6[i, :, :, 0] = pyrDown(y5[i, :, :, 0])

    x = x.astype('float32')
    x /= 255

    y = [y6, y5, y4, y3, y2, y1]

    return (x, y)


def data_generator(index_org, isTrain = True, isGAN = True, close_far_all = 0, batchSize = 10):
    image_mean = np.zeros(shape=(448, 640, 3))
    image_mean[:,:,0] = 114*np.ones(shape=(448, 640))
    image_mean[:,:,1] = 105*np.ones(shape=(448, 640))
    image_mean[:,:,2] = 97*np.ones(shape=(448, 640))

    index_all = sio.loadmat('train_val_test_index.mat');

    if isTrain:
        path = data_path
        index = index_all['total'].transpose()
        shuffle(index)
    else:
        index = index_all['total'].transpose()
        path = data_path
        shuffle(index)

    i = 0
    while(True):
        if isGAN:
            yield loadDataGAN(index, i, batchSize, path, image_mean)
        else:
            if close_far_all == 0:
                yield loadData(index, i, batchSize, path, image_mean)
            elif close_far_all == 1:
                yield loadData_close(index, i, batchSize, path, image_mean)
            elif close_far_all == 2:
                yield loadData_far(index, i, batchSize, path, image_mean)
            elif close_far_all == 3:
                (x, y_close) = loadData_close(index, i, batchSize, path, image_mean)
                (x, y_far) = loadData_far(index, i, batchSize, path, image_mean)
                (x, y) = loadData(index, i, batchSize, path, image_mean)
                yield (x, [y_close[0], y_close[1], y_close[2], y_close[3], y_close[4], y_close[5],
                           y_far[0], y_far[1], y_far[2], y_far[3], y_far[4], y_far[5],
                           y[0], y[1], y[2], y[3], y[4], y[5]])
            elif close_far_all == 4:
                yield loadData_edge(index, i, batchSize, path, image_mean)
            elif close_far_all == 5:
                yield loadData_fill_hole_run(index, i, batchSize, path, image_mean)
            elif close_far_all == 6:
                yield loadData_predict_hole(index, i, batchSize, path, image_mean)
                

        i = i + batchSize



def zero_mask(y):
    return tf.to_float(K.not_equal(K.zeros_like(y), y))

def zero_mask_inv(y):
    return tf.to_float(K.equal(K.zeros_like(y), y))

def my_loss(y_true, y_pred):
    return K.mean(tf.multiply(K.square(y_pred - y_true), tf.to_float(K.not_equal(K.zeros_like(y_true), y_true))))

def metric_L1_real(y_true, y_pred):
    return K.mean(tf.realdiv(tf.multiply(K.abs(y_pred-y_true), zero_mask(y_true)), tf.add(y_true, zero_mask_inv(y_true))))

def metric_L1_inv(y_true, y_pred):
    return K.mean(K.abs(tf.realdiv(zero_mask(y_true), y_pred) - tf.realdiv(zero_mask(y_true), tf.add(y_true, zero_mask_inv(y_true)))))

def copy_weights(model, d):
    for i in range(6):
        model_layer_name = model._feed_output_names[i+6]
        for j in range((len(d[i].layers) - 1), 0):
            model.get_layer(name=model_layer_name).set_weights(d[i].layers[j].get_weights())
            #update the layer name in model
            model_layer_name = model.get_layer(name=model_layer_name).input.name.split('/')[0]

    return True

def pyrDown(y):
    y_down = y[::2, ::2]

    ind0 = y[:,:]==0
    mask = np.ones_like(y)
    mask[ind0] = 0

    y_pyr = cv2.pyrDown(y)
    mask_pyr = cv2.pyrDown(mask)

    ind0_down = y_down[:,:]==0
    mask_pyr[ind0_down] = None
    y_pyr = np.true_divide(y_pyr, mask_pyr)

    mask_nan = np.isnan(y_pyr)
    ind_nan = mask_nan[:,:]==True
    y_pyr[ind_nan] = 0

    return y_pyr





