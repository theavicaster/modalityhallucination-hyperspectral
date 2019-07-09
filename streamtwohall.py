import scipy.io as sio
import numpy as np

import keras
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config.gpu_options.per_process_gpu_memory_fraction = 0.05
set_session(tf.Session(config=config))

import keras.backend as K
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

X_one = (X[:,:,:,0:100])
X_two = (X[:,:,:,100:200])

from sklearn.model_selection import train_test_split

(Xtrain_one, Xtest_one, Ytrain, Ytest) = train_test_split(X_one, Y, random_state = 666, test_size = 0.75)
(Xtrain_two, Xtest_two, Ytrain, Ytest) = train_test_split(X_two, Y, random_state = 666, test_size = 0.75)

from keras.models import Model
from keras.models import load_model
from keras import optimizers
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Input, 
                          concatenate, Add, BatchNormalization, Dropout, Lambda)

streamtwo = load_model('/home/SharedData/Avinandan/DarrellHallucination/streamtwo.h5')

streamtwofeatures = Model(inputs=streamtwo.get_layer('Indian_Pines_Input_2').input,
                       outputs=streamtwo.get_layer('Indian_Pines_Output_2').output)
print(streamtwofeatures.summary())

modelinput1 = Input(shape=(15, 15, 100), name='Indian_Pines_Input_1')
modelinput2 = Input(shape=(15, 15, 100), name='Indian_Pines_Input_2')
conv11 = Conv2D(128, (5, 5), activation='relu')(modelinput1)

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
model1 = Dense(200, activation='sigmoid', name='Indian_Pines_Output_2_Hallucinated')(model1)
model1hall = model1

'''

combinedoutput= Dense(1024,activation='relu')(model1)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.1)(combinedoutput)

combinedoutput= Dense(1024,activation='relu')(combinedoutput)
combinedoutput = BatchNormalization()(combinedoutput)
combinedoutput = Dropout(0.1)(combinedoutput)

#Set temperature parameter for softmax
combinedoutput = Lambda(lambda x: x / 2)(combinedoutput)
combinedoutput= Dense(16,activation='softmax', name='Classifier_Output_1_Hallucinated')(combinedoutput)

'''


for layer in streamtwo.layers[18:]:
    model1 = layer(model1)


original = streamtwofeatures(modelinput2)

model = Model(inputs=[modelinput1, modelinput2], outputs=model1)

print(model.summary())

for n in range(18,26):
	model.layers[n].trainable = False

for l in model.layers:
    print(l.name, l.trainable)

def custom_loss(hall,orig):
    def custom(y_true, y_pred):
        return (keras.losses.categorical_crossentropy(y_true, y_pred)
                + l2_loss(hall,orig))
    return custom

def l2_loss(hall,orig):

	return K.sqrt(K.sum((hall - orig)**2))

model_loss = custom_loss(hall=model1hall, orig=original)


sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
model.compile(optimizer=sgd, loss=model_loss, metrics=['accuracy'])




model_json = model.to_json()
with open("/home/SharedData/Avinandan/DarrellHallucination/streamtwohall.json", "w") as json_file:
    json_file.write(model_json)


checkpoint = ModelCheckpoint('/home/SharedData/Avinandan/DarrellHallucination/streamtwohall.hdf5', monitor='val_acc', verbose=1,save_best_only=True, mode='max')
callback_list = [checkpoint]

model.fit(
    [Xtrain_one,Xtrain_two], Ytrain, callbacks=callback_list,
    validation_data=([Xtest_one,Xtest_two],Ytest),
    epochs=300, verbose = 2)









