import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage import transform
from skimage.transform import resize, rotate
from skimage.morphology import label

from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
BATCH_SIZE = 10
TRAIN_PATH = 'training_images'
TRAIN_MASK_PATH = 'training_masks'
TEST_PATH = 'testing_images'
VALID_PATH = 'validation_images'
VALID_MASK_PATH = 'validation_masks'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


train_ids = next(os.walk(TRAIN_PATH))[2]
train_mask_ids = next(os.walk(TRAIN_MASK_PATH))[2]
valid_ids = next(os.walk(VALID_PATH))[2]
valid_mask_ids = next(os.walk(VALID_MASK_PATH))[2]
print('Getting and resizing train/validation images and masks ... ')


def getImages(ids_, PATH):
	X_train = np.zeros((len(ids_), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	for i in range(len(ids_)):
		id_ = ids_[i]
		img = imread(PATH + '/' + id_)[:,:,:IMG_CHANNELS]
		img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		X_train[i] = img
	print('Finished getting images from ' + PATH + ' with size ' + str(len(X_train)))
	return X_train

def getMaskImages(ids_, PATH):
    Y_train = np.zeros((len(ids_), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    for i in range(len(ids_)):
        id_ = ids_[i]
        mask = imread(PATH + '/' + id_)
        mask = np.array(mask, dtype=np.uint8)
        mask = np.round(mask/255)
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
            preserve_range=True), axis=-1)
        Y_train[i] = mask
    print('Finished getting masks from ' + PATH + ' with size ' + str(len(Y_train)))
    return Y_train

X_train = getImages(train_ids, TRAIN_PATH)
Y_train = getMaskImages(train_mask_ids, TRAIN_MASK_PATH)
X_valid = getImages(valid_ids, VALID_PATH)
Y_valid = getMaskImages(valid_mask_ids, VALID_MASK_PATH)

print('Done!')

def addMoreFlippedImages(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    
    for i in range(0, imgs.shape[0]):

        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
    
        av = a[::-1, :]
        ah = a[:, ::-1]
        bv = b[::-1, :]
        bh = b[:, ::-1]
        cv = c[::-1, :]
        ch = c[:, ::-1]
    
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
    
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
    more_images = np.concatenate((imgs,v,h))
    
    return more_images

def addMoreFlippedMasks(mask):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, mask.shape[0]):
        a=mask[i,:,:,0]
        
        av = a[::-1, :]
        ah = a[:, ::-1]
        
        vert_flip_imgs.append(av.reshape(IMG_WIDTH,IMG_WIDTH,1))
        hori_flip_imgs.append(ah.reshape(IMG_WIDTH,IMG_WIDTH,1))
        
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_mask = np.concatenate((mask,v,h))

    return more_mask


X_train = addMoreFlippedImages(X_train)
Y_train = addMoreFlippedMasks(Y_train)

print(X_train.shape, Y_train.shape)

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

def unet():
    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

model = unet()
model.summary()

# opt = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

opt = Adam()

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mean_iou])
print('Compiled the model')

model.summary()

checkpoint_path = 'model/model_best_best.h5'

earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=5, 
                              verbose=1, 
                              min_delta = 0.0001,
                              mode='min')

modelCheckpoint = ModelCheckpoint(checkpoint_path,
                                  monitor = 'val_loss', 
                                  save_best_only = True, 
                                  mode = 'min', 
                                  verbose = 1,
                                  save_weights_only = False)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.1, 
                                    patience=3, 
                                    verbose=1,
                                    min_lr=0.0001,
                                    min_delta=0.0001)

callbacks_list = [modelCheckpoint, earlyStopping, reduceLROnPlat]

model.fit(
	X_train,
    Y_train,
	validation_data = (X_valid, Y_valid),
    batch_size = BATCH_SIZE,
	epochs=50,
    callbacks=callbacks_list,
    shuffle = True)
print('Finished Training!')

model.save('model/current_model.h5')
print('Saved current model')