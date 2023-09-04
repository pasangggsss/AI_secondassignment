"21422019"
TrainingImagePath='./photo'

from keras.preprocessing.image import ImageDataGenerator as IMG

"""
creating a datagenerator object for generating the images in runtime from a directory. 
photo/ directory contains all the images required for the model
"""

mdl_train_data_gen_all = IMG(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=20,
        vertical_flip=True,
        channel_shift_range=0.1,
        horizontal_flip=True
        )

mdl_test_data_gen_all = IMG()

mdl_data_set_train_data = mdl_train_data_gen_all.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        )

mdl_data_set_test_data = mdl_test_data_gen_all.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        )

mdl_data_set_test_data.class_indices

cls_trn_mdl=mdl_data_set_train_data.class_indices

"""
saving the classMap of the DATASET

"""
rm={}
for v,n in zip(cls_trn_mdl.values(),cls_trn_mdl.keys()):
    rm[v]=n
 
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(rm, fileWriteStream)
 
OutputNeurons=len(rm)
from keras.models import Sequential
from keras.layers import Convolution2D as conv2
from keras.layers import MaxPool2D as pool2
from keras.layers import Flatten
from keras.layers import Dense, Dropout
class_trained_mdl_predict= Sequential()


"""
Model architecture: 
[
        conv2 3,3
        conv2 3,3
        maxpool-2,2
        conv2 7,7
        pool2 2,2
        Flatten
        Dropout 0.7
        Dense
        Dense
]

"""

class_trained_mdl_predict = Sequential();
class_trained_mdl_predict.add(conv2(32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
class_trained_mdl_predict.add(conv2(64, kernel_size=(3, 3), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
class_trained_mdl_predict.add(pool2(pool_size=(2,2)))
class_trained_mdl_predict.add(conv2(64, kernel_size=(7, 7), strides=(1, 1), activation='relu'))
class_trained_mdl_predict.add(pool2(pool_size=(2,2)))
class_trained_mdl_predict.add(conv2(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
class_trained_mdl_predict.add(pool2(pool_size=(2,2)))
class_trained_mdl_predict.add(Flatten())
class_trained_mdl_predict.add(Dropout(0.7))
class_trained_mdl_predict.add(Dense(64, activation='relu'))
class_trained_mdl_predict.add(Dense(OutputNeurons, activation='softmax'))
class_trained_mdl_predict.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])


class_trained_mdl_predict.fit_generator(
                    mdl_data_set_train_data,
                    epochs=100,
                    validation_data=mdl_data_set_test_data,
                    validation_steps=7)

class_trained_mdl_predict.save_weights('./model_final.h5')

