import os

from src.settings import SEED_VALUE
# os.environ['PYTHONHASHSEED']=str(SEED_VALUE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.random.set_seed(SEED_VALUE)

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


