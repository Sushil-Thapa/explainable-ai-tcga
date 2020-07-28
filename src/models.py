import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


tf.keras.backend.clear_session() # reset keras session

def get_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim = input_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(output_dim, activation="softmax"))
    model.summary()
    return model


