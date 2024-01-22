import tensorflow as tf
from pathlib import Path

from keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from keras import backend as K
from keras.optimizers import SGD
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
        self.numlayers =  hyperparameters.get(NUMLAYERS_LABEL)
        self.datadescr =  hyperparameters.get(DATADESCR_LABEL)

    def getFileNamePart(self):
        filenamePart = TWOUNDERSCORE + NUMLAYERS_LABEL+ UNDERSCORE + str(self.numlayers)
        filenamePart = filenamePart + TWOUNDERSCORE + NUNITS_LABEL + UNDERSCORE + str(self.nunits)
        filenamePart = filenamePart + TWOUNDERSCORE + EPOCHS_LABEL + UNDERSCORE + str(self.epochs)
        filenamePart = filenamePart + TWOUNDERSCORE + DROPOUT_LABEL + UNDERSCORE + str(self.dropout)
        filenamePart = filenamePart + TWOUNDERSCORE + DATADESCR_LABEL + UNDERSCORE + str(self.datadescr)
        filenamePart = filenamePart + TWOUNDERSCORE + NUMFEATURES_LABEL + UNDERSCORE + str(self.numfeatures)
        return filenamePart


class ModelManager:

    def __init__(self,sensor,hyperparams,reTrain=False,dirModelli='modelli'):
        self.__isTrained = False
        self.__hyperparams = HyperParameters(hyperparams)
        self.__filename = self.__composeFileName(sensor,dirModelli)
        self.__model = self.__initModel(reTrain)

    def __generateModel(self):
        my_model = Sequential()
        my_model.add(LSTM(units=self.__hyperparams.nunits, return_sequences=True,
                          input_shape=(self.__hyperparams.timesteps, self.__hyperparams.numfeatures)))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(LSTM(units=self.__hyperparams.nunits, return_sequences=True))
        my_model.add(Dropout(self.__hyperparams.dropout))
        my_model.add(Dense(units=1))
        CHOSEN_OPTIMIZATION = SGD(learning_rate=0.01)
        # CHOSEN_LOSS = 'mean_squared_error'
        my_model.compile(optimizer=CHOSEN_OPTIMIZATION, loss=CHOSEN_LOSS, metrics=[MAPE_LABEL, MAE_LABEL, MSE_LABEL])

        return my_model

    def __initModel(self,reTrain):
        my_file = Path(self.__filename)
        if my_file.is_file():
            my_model = tf.keras.models.load_model(self.__filename)
            self.__isTrained = True
            if reTrain:
                self.__isTrained = False
            return my_model
        else:
            my_model = self.__generateModel()
            return my_model

    def __composeFileName(self,sensor,dirModelli):
        filename = dirModelli + '/'+ LSTM_MODEL_LABEL + UNDERSCORE + SENSOR_LABEL + UNDERSCORE + str(sensor) + UNDERSCORE + self.__hyperparams.getFileNamePart()
        filename = filename + SUFFISSO_MODELLO_KERAS
        return filename

    def trainModel(self,x_train,y_train):
        history = self.__model.fit(x_train, y_train, epochs=self.__hyperparams.epochs, batch_size=self.__hyperparams.batch_size,
                               validation_split=self.__hyperparams.validation_split)
        self.__model.save(self.__filename)
        self.__isTrained = True

    def getModel(self):
        return self.__model

    def getFileNamePartFromHyperParams(self):
        return self.__hyperparams.getFileNamePart()

    def isModelTrained(self):
        return self.__isTrained
