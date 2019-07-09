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
from keras.models import model_from_json
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


def custom_loss(hall,orig):
    def custom(y_true, y_pred):
        return (keras.losses.categorical_crossentropy(y_true, y_pred)
                + l2_loss(hall,orig))
    custom.__name__ = "Custom Loss"
    return custom



def l2_loss(hall,orig):

	return K.sqrt(K.sum((hall - orig)**2))



from keras.models import load_model
from keras import optimizers
from keras.models import model_from_json

# load json and create model
json_file = open('/home/SharedData/Avinandan/DarrellHallucination/streamonehall.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
streamonehall = model_from_json(loaded_model_json)
# load weights into new model
streamonehall.load_weights("/home/SharedData/Avinandan/DarrellHallucination/streamonehall.hdf5")



streamone = load_model('/home/SharedData/Avinandan/DarrellHallucination/streamone.h5')
#streamonehall = load_model('/home/SharedData/Avinandan/DarrellHallucination/streamonehall.h5') #custom_objects={ custom: custom_loss(hall,orig)} )

sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
streamone.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
streamonehall.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

results = streamone.evaluate(Xtest_one,Ytest, batch_size=32)
resultshall = streamonehall.evaluate(Xtest_two,Ytest, batch_size=32)

print(" ")
print('ORIGINAL NETWORK test loss, test acc:', results)
print('HALLUCINATED test loss, test acc:', resultshall)




