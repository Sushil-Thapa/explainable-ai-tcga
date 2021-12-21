import os,sys
import random
import argparse
import time
import json

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.feather as feather

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn import metrics

def get_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='List of options to train and explain TCGA', add_help=False)

    # Add the arguments
    parser.add_argument('-t',
                        "--train",
                        action='store_true',
                        help='Optional train flag')

    parser.add_argument('-e',
                        "--explain",
                        action='store_true',
                        help='Optional lime explain flag')

    parser.add_argument('-n',
                        '--num_instances', 
                        action='store', 
                        type=int, 
                        default=30,
                        help='number of instances to explain from train and test set each')

    parser.add_argument('-m',
                        '--num_models', 
                        action='store', 
                        type=int, 
                        default=10,
                        help='number of models to train')

    parser.add_argument('-p',
                        '--num_samples', 
                        action='store', 
                        type=int, 
                        default=10000,
                        help='number of perturbations in LIME')

    parser.add_argument('-s',
                        '--seed', 
                        action='store', 
                        type=int, 
                        default=42,
                        help='seed value')


    parser.add_argument('-f',
                        "--save_format",
                        choices=['json','html','pdf'],
                        default='pdf',
                        help='Save option, saves to out/ folder')

    parser.add_argument('-c',
                        "--class_name",
                        default='brain',
                        help='class names')

    # Execute the parse_args() method
    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    return args 

args = get_args()

random_state = int(args.seed)
np.random.seed(random_state)
random.seed(random_state)
tf.random.set_seed(random_state)

print("Trainig mode: ",args.train)

# number of test set sample(tcga) for inspection
n_tcga_test = int(args.num_instances)

# Load dataset


print(args)


def get_dataset():
    print("loading dataset")
    
    #df = feather.read_feather('../../data/ben/aug/june18_TCGA.NMJ123.log.RDS.csv.feather')
    # backup = df.copy()
    filename = "../data/june18_TCGA.NMJ123.log.RDS.csv"
    # df = pd.read_csv(filename)
    # feather.write_feather(df, filename+".feather")

    df = feather.read_feather(filename+".feather")
    print("dataset loaded.. preprocessing...")

    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(index='X')

    df = df.reset_index()
    df = df.rename(columns={'index':'Y'})

    df['Y'] = df['Y'].apply(lambda x: x.split(".")[0])
    # df['Y'].value_counts().plot(kind='barh', figsize = (10,6))

    label_encoder = LabelEncoder()
    df['Y'] = label_encoder.fit_transform(df['Y'])
    df['Y'].head()

    print(len(label_encoder.classes_))

    Y = df.pop("Y")
    Y = tf.keras.utils.to_categorical(Y)
    print("returninig dataset")
    return df, Y, label_encoder


df, Y, label_encoder = get_dataset()

"""### Train-Test Split (Stratified and shuffled)"""
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size = 0.2, random_state = random_state, stratify=Y, shuffle=True)
print("Training and validation dataset shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


models_list = {}

"""### Model Definition& Config"""
tf.keras.backend.clear_session() # reset keras session
model = Sequential()
model.add(Dense(1024, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(y_train.shape[1], activation="softmax"))
model.summary()

## Add model training configs
checkpoint_path = f"out/cp.ckpt"

if not os.path.exists('out'):
    os.makedirs('out')

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor='val_accuracy',
                                                verbose=1)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['categorical_crossentropy','accuracy'])

if args.train:  # just load model and get explanations if not training
    """## Training"""
    epochs = 10
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        verbose=1, 
                        epochs=epochs, 
                        shuffle=True,
                        callbacks = [ es_callback, cp_callback])
else:
    # The model weights (that are considered the best) are loaded into the model.
    print("Loading model weights...")
    model.load_weights(checkpoint_path)

def get_saliency_map(model, image, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        
        
        max_outp = predictions[:, class_idx]
    
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(max_outp, image)
    
    # # uncomment to take maximum across channels 
    # gradient = tf.reduce_max(gradient, axis=-1)
    
    # convert to numpy
    gradient = gradient.numpy()
    
    # normaliz between 0 and 1
    # min_val, max_val = np.min(gradient), np.max(gradient)
    # smap = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())
    
    return gradient

def saliency_map(output, input, name="saliency_map"):
    """
    Produce a saliency map as described in the paper:
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/abs/1312.6034>`_.
    The saliency map is the gradient of the max element in output w.r.t input.
    Returns:
        tf.Tensor: the saliency map. Has the same shape as input.
    """
    max_outp = tf.reduce_max(output, 1)
    saliency_op = tf.gradients(max_outp, input)[:][0]
    return tf.identity(saliency_op, name=name)


class_idx = list(label_encoder.classes_).index(str(args.class_name))  # axis=1 applies on and remove classinfo
X_brain_test = X_test[np.argmax(y_test, axis=1) == np.ones_like(np.argmax(y_test, axis=1)) * class_idx] # Brain has index 1 so ones_like * index
import ipdb; ipdb.set_trace()

# out = model(X_brain_test)

smap = get_saliency_map(model, tf.convert_to_tensor(X_brain_test, dtype=tf.float32), class_idx)

smap = get_saliency_map(model, tf.convert_to_tensor(X_brain_test, dtype=tf.float32), class_idx)
smap = get_saliency_map(model, tf.convert_to_tensor(X_brain_test, dtype=tf.float32), class_idx)
smap = get_saliency_map(model, tf.convert_to_tensor(X_brain_test, dtype=tf.float32), class_idx)


arr = smap.copy()
x = arr.shape[0]
N = 50
top_indices = np.argsort(arr, axis=1)[:, -N:]
topn_maps = arr[np.repeat(np.arange(x), N), top_indices.ravel()].reshape(x, N)
np.savetxt(f"top{N}_smap_idxs.csv", top_indices.astype(int),  fmt='%i', delimiter=',' )
# np.savetxt("topN_smap_idxs.csv",  np.argpartition(smap, np.argmin(smap, axis=0))[:, -100:].astype(int),  fmt='%i', delimiter=',')
# a[np.repeat(np.arange(x), 3), ind.ravel()].reshape(x, 3)