import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from keras.regularizers import l2

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TEST_PATH = 'testing_images'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

test_ids = next(os.walk(TEST_PATH))[2]

print('Getting and resizing train images and masks ... ')

def getTestImages(ids_, PATH):
	X_train = np.zeros((len(ids_), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	sizes_test = []
	for i in range(len(ids_)):
		id_ = ids_[i]
		img = imread(PATH + '/' + id_)[:,:,:IMG_CHANNELS]
		sizes_test.append([img.shape[0], img.shape[1]])
		img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		X_train[i] = img
	print('Finished getting from ' + PATH + ' with size ' + str(len(X_train)))
	return X_train, sizes_test

X_test, sizes_test = getTestImages(test_ids, TEST_PATH)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

model = load_model('model/model_best_best.h5', custom_objects={'mean_iou':mean_iou})
print('Model loaded')


preds_test = model.predict(X_test, verbose=1)
print('Finished Predicting Model')

preds_test_t = (preds_test > 0.5).astype(np.uint8)

preds_test_upsampled = []
for i in range(len(preds_test)):
	preds_test_upsampled.append(resize(
		np.squeeze(preds_test[i]),
		(sizes_test[i][0], sizes_test[i][1]),
		mode='constant',
		preserve_range=True))
print('Finished Resizing Predicted Mask')

def getTestID(test_ids):
	trim_ids = []
	for i in range(len(test_ids)):
		trim_ids.append(test_ids[i][:-4])
	return trim_ids

TEST_MASK_PATH = 'testing_masks'

def savePrediction2Images(pred):
	for i in range(len(pred)):
		filename = test_ids[i].replace('img', 'mask')
		imsave(os.path.join(TEST_MASK_PATH, filename), pred[i])

savePrediction2Images(preds_test_upsampled)

# encoding function
# based on the implementation: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python/code
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# get a sorted list of all mask filenames in the folder
masks = [f for f in os.listdir(TEST_MASK_PATH) if f.endswith('.jpg')]
masks = sorted(masks, key=lambda s:int(s.split('_')[2].split('.')[0]))

# encode all masks
encodings = []
for file in masks:
    mask = imread(os.path.join(TEST_MASK_PATH, file))
    #img_size =10
    #mask = resize(mask, (img_size, img_size), mode='constant', preserve_range=True)
    mask = np.array(mask, dtype=np.uint8)
    mask = np.round(mask/255)
    encodings.append(rle_encoding(mask))


# (** update) the path where to save the submission csv file
sub = pd.DataFrame()
sub['ImageId'] = pd.Series(masks).apply(lambda x: os.path.splitext(x)[0])
sub['EncodedPixels'] = pd.Series(encodings).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('submission1.csv', index=False)
