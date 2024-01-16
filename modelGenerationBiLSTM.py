import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Bidirectional
from keras import backend as K
from keras.optimizers import SGD, Adam
from constants import *
import sys

class HyperParameters:

    def __init__(self,hyperparameters):
        self.timesteps = hyperparameters.get(TIME_LAG_LABEL)
        self.numfeatures = hyperparameters.get(NUMFEATURES_LABEL)
        self.dropout = hyperparameters.get(DROPOUT_LABEL)
        self.nunits = hyperparameters.get(NUNITS_LABEL)
        self.batch_size = hyperparameters.get(BATCH_SIZE_LABEL)
        self.validation_split = hyperparameters.get(VALIDATION_SPLIT_LABEL)
        self.epochs = hyperparameters.get(EPOCHS_LABEL)

class ModelManager:

    def __init__(self,filename,hyperparams):
        self.__isTrained = False
        self.__filename = filename
        self.__hyperparams = HyperParameters(hyperparams)
        self.__model = self.__initModel()

    def __initModel(self):
        my_file = Path(self.__filename)
        if my_file.is_file():
            my_model = tf.keras.models.load_model(self.__filename)
            self.__isTrained = True
            return my_model
        else:
            my_model = self.__generateModel()
            return my_model

    def trainModel(self,x_train,y_train):
        history = self.__model.fit(x_train, y_train, epochs=self.__hyperparams.epochs, batch_size=self.__hyperparams.batch_size,
                               validation_split=self.__hyperparams.validation_split)
        self.__model.save(self.__filename)
        self.__isTrained = True

    def getModel(self):
        return self.__model

    def isModelTrained(self):
        return self.__isTrained


    def __generateModel(self):
        my_model = Sequential()
        my_model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(self.__hyperparams.timesteps, self.__hyperparams.numfeatures))))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Dense(units=1))
        custom_optimizer = Adam(learning_rate=0.01)
        my_model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        return my_model

    '''
        def __generateModel(self):
            my_model = Sequential()
            my_model.add(Bidirectional(LSTM(units=self.__hyperparams.nunits, return_sequences=True,
                          input_shape=(self.__hyperparams.timesteps, self.__hyperparams.numfeatures))))
            my_model.add(Dropout(self.__hyperparams.dropout))
            my_model.add(Bidirectional(LSTM(units=self.__hyperparams.nunits, return_sequences=False)))
            my_model.add(Dropout(self.__hyperparams.dropout))
            my_model.add(Dense(units=1))
            CHOSEN_OPTIMIZATION = SGD(learning_rate=0.01)
            # CHOSEN_LOSS = 'mean_squared_error'
            my_model.compile(optimizer=CHOSEN_OPTIMIZATION, loss=CHOSEN_LOSS, metrics=[MAPE_LABEL, MAE_LABEL, MSE_LABEL])

            return my_model
    '''