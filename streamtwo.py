import scipy.io as sio
import numpy as np

import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config.gpu_options.per_process_gpu_memory_fraction = 0.05
set_session(tf.Session(config=config))

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
np.random.seed(666)

def loadIndianPinesData():
   
    data = sio.loadmat('/home/SharedData/Avinandan/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = sio.loadmat('/home/SharedData/Avinandan/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']
    
    return data, labels

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=17, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels



data, indian_pines_gt = loadIndianPinesData()

data = (data - data.mean(axis=(0, 1), keepdims=True))/(data.std(axis=(0, 1), keepdims=True))
data = data.astype(np.float32)
print(data.dtype)

X, Y = createPatches(data, indian_pines_gt, windowSize=15)

print(X.shape,Y.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

print(X.shape,Y.shape)

X_two = (X[:,:,:,100:200])

from sklearn.model_selection import train_test_split

(Xtrain_two, Xtest_two, Ytrain, Ytest) = train_test_split(X_two, Y, random_state = 666, test_size = 0.75)

from keras.models import Model
from keras import optimizers
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Input, 
                          concatenate, Add, BatchNormalization, Dropout, Lambda)

modelinput2 = Input(shape=(15, 15, 100), name='Indian_Pines_Input_2')
conv11 = Conv2D(128, (5, 5), activation='relu')(modelinput2)

conv2 = Conv2D(128, (1, 1), activation='relu', padding='same')

model1 = conv2(conv11)

bn2 = BatchNormalization()
model1 = bn2(model1)

conv3 = Conv2D(128, (1, 1), activation='relu', padding='same')
model1 = conv3(model1)

bn3 = BatchNormalization()
model1 = bn3(model1)

conv4 = Conv2D(128, (1, 1), activation='relu', padding='same')
model1 = conv4(model1)

bn4 = BatchNormalization()
model1 = bn4(model1)

conv5 = Conv2D(128, (1, 1), activation='relu', padding='same')
model1 = conv5(model1)

bn5 = BatchNormalization()
model1 = bn5(model1)

model1 = Conv2D(64, (1, 1), activation='relu')(model1)
model1 = BatchNormalization()(model1)

model1 = Conv2D(64, (1, 1), activation='relu')(model1)
model1 = BatchNormalization()(model1)
model1 = Dropout(0.1)(model1)

model1 = Flatten()(model1)
model1 = Dense(512, activation='relu')(model1)
model1 = Dense(200, activation='sigmoid', name='Indian_Pines_Output_2')(model1)

combinedoutput= Dense(1024,activation='relu', name = 'Classifier_Input_2')(model1)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.1)(combinedoutput)

combinedoutput= Dense(1024,activation='relu')(combinedoutput)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.1)(combinedoutput)

#Set temperature parameter for softmax
combinedoutput = Lambda(lambda x: x / 2)(combinedoutput)
combinedoutput= Dense(16,activation='softmax', name='Classifier_Output_2')(combinedoutput)


model = Model(inputs=modelinput2, outputs=combinedoutput)
sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('/home/SharedData/Avinandan/DarrellHallucination/streamtwo.h5', monitor='val_acc', verbose=1,save_best_only=True, mode='max')
callback_list = [checkpoint]

model.fit(
    Xtrain_two, Ytrain, callbacks=callback_list,
    validation_data=(Xtest_two,Ytest),
    epochs=300, verbose = 2)














