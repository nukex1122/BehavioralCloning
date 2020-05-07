import os
import csv
import math 
import random 

import tensorflow as tf
import cv2
import numpy as np
import sklearn

from pathlib import PurePosixPath
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D, Cropping2D, MaxPool2D, BatchNormalization, Activation
from keras.layers import Activation

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

samples = []

with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        center_angle = float(line[3])
        if (center_angle==0.0 and random.random()>0.8) or center_angle!=0.0:
            samples.append(line)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    shuffle(samples)
    
    while 1:
        for i in range(0, num_samples, batch_size):
            batch_samples = samples[i: i+batch_size]
            images, labels = [], []
            
            for batch in batch_samples:
                name = './data/IMG/'+batch[0].split('/')[-1]
                
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch[3])
                images.append(center_image)
                labels.append(center_angle)
                images.append(cv2.flip(center_image,1))
                labels.append(-1*center_angle)

                name = './data/IMG/'+batch[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                left_angle = float(batch[3]) + .2+ .05 * np.random.random()
                images.append(left_image)
                labels.append(left_angle)
                images.append(cv2.flip(left_image,1))
                labels.append(-left_angle)

                name = './data/IMG/'+batch[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                right_angle = float(batch[3]) - .2 - .05 *np.random.random()
                images.append(right_image)
                labels.append(right_angle)
                images.append(cv2.flip(right_image,1))
                labels.append(-1*right_angle)

            X_train = np.array(images)
            y_train = np.array(labels)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size=64

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((65,20), (0,0))))


model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="elu"))
model.add(Convolution2D(64, (3, 3), activation="elu"))
model.add(Convolution2D(64, (3, 3), activation="elu"))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

optimizer = Adam(lr=0.002)
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)
                                     
model.save('model.h5')

print('Model Saved!')

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()


print(model.summary())

