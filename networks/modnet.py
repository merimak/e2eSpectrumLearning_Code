# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:54:00 2017

@author: mkulin
"""

import os
os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
import theano as th
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l1, l2
from keras import optimizers
from keras import backend as K
K.set_image_dim_ordering('th')
import keras


class ModNet2:
    
    """This is the CNN2 network for radio modulation classification
    """
    def __init__(self, lr, dr):
        self.lr = lr
        self.dr= dr
        self.model=models.Sequential()
        self.trainPerf={}
    
    def build(self, in_shp, classes, weightsPath=None):
        #initialize the model
        self.model.add(Reshape([1]+in_shp, input_shape=in_shp))
        self.model.add(ZeroPadding2D((0, 1)))
        #Add convolutional layers		
        self.model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
        self.model.add(Dropout(self.dr))
        self.model.add(ZeroPadding2D((0, 1)))
        self.model.add(Convolution2D(80, 2, 3, border_mode='valid', activation="relu", name="conv2", init='glorot_uniform'))
        self.model.add(Dropout(self.dr))
        self.model.add(Flatten())
        #Add dense layer
        self.model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
        self.model.add(Dropout(self.dr))
        self.model.add(Dense(len(classes), init='he_normal', name="dense2"))
        self.model.add(Activation('softmax'))
        self.model.add(Reshape([len(classes)]))  
        adam= optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam)
        self.model.summary()
        
        if weightsPath is not None:
            self.model.load_weights(weightsPath)
            # return the constructed network architecture

    def train(self, X_train, Y_train, X_test, Y_test, nb_epoch=50, batch_size=1024, basepath=""):
        #path=basepath+'_e'+str(nb_epoch)+'_dr'+str(self.dr)+'_lr'+str(self.lr)
        path=basepath
        model_filepath=path + ".weights.h5"
        history_filepath=path + ".history.dat"
        #Train and save the model
        history = self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, Y_test),
        callbacks = [
            keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        ])
        
        #Save the training progress per epochs
        self.trainPerf['epochs']=history.epoch
        self.trainPerf['tr_loss']=history.history['loss']
        self.trainPerf['val_loss']=history.history['val_loss']
        import pickle
        with open(history_filepath, "wb") as f:
            pickle.dump(self.trainPerf, f)
            f.close()


