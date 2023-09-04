import cv2
import pickle

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten, Dropout
from keras.layers import Dense
f = open('./ResultsMap.pkl', 'rb')
l = pickle.load(f)
f.close()
neuron_finalized = len(l)
def loadModel(neuron_finalized):
    class_trained_mdl_predict = Sequential();
    class_trained_mdl_predict.add(Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
    class_trained_mdl_predict.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
    class_trained_mdl_predict.add(MaxPool2D(pool_size=(2,2)))
    class_trained_mdl_predict.add(Convolution2D(64, kernel_size=(7, 7), strides=(1, 1), activation='relu'))
    class_trained_mdl_predict.add(MaxPool2D(pool_size=(2,2)))
    class_trained_mdl_predict.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    class_trained_mdl_predict.add(MaxPool2D(pool_size=(2,2)))

    class_trained_mdl_predict.add(Flatten())
    class_trained_mdl_predict.add(Dropout(0.7))
    class_trained_mdl_predict.add(Dense(64, activation='relu'))
    class_trained_mdl_predict.add(Dense(neuron_finalized, activation='softmax'))
    class_trained_mdl_predict.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

    return class_trained_mdl_predict

class_trained_mdl_predict = loadModel(neuron_finalized)

class_trained_mdl_predict.load_weights('./model_final.h5');

import os

folder_for_test_img = "./testtin/";
f = open('./ResultsMap.pkl', 'rb')
l = pickle.load(f)

from keras.utils import load_img
import numpy as np

total = 0;
successful = 0;
failure = [];


"""
lOOPING OVER ALL THE FILES INSIDE TESTTIN FOLDER.
GOES THROUGH EACH CLASS (BY FOLDER)
CHECCKS IF THE RESULT MATCHES THE CLASSNAME.
IF NOT THEN IT IS ADDED TO FAILURE LIST

"""
for i in os.listdir(folder_for_test_img):
    for j in os.listdir(folder_for_test_img + i):
        total +=1
        ImagePath = folder_for_test_img + i+"/" + j

        test_image=load_img(ImagePath,target_size=(64, 64))
        test_image=np.expand_dims(test_image,axis=0)
        result=class_trained_mdl_predict.predict(test_image,verbose=0)
        result = l[np.argmax(result)]
        print(i)
        if(result == i):
            successful = successful+ 1
        else: failure.append(i)

        print('Model is predicting the result as: ', result)


print(f"{total=}\n {successful=}\n {failure=} ")


