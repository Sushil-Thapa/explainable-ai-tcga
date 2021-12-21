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

import shap
shap.initjs()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn import metrics

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
args = parser.parse_known_args()[0]

random_state = int(args.seed)
np.random.seed(random_state)
random.seed(random_state)
tf.random.set_seed(random_state)

print("Trainig mode: ",args.train)

# number of test set sample(tcga) for inspection
n_tcga_test = int(args.num_instances)

# Load dataset

args = parser.parse_args()
print(args)


def get_dataset():
    print("loading dataset")
    df = pd.read_csv("../data/june18_TCGA.NMJ123.log.RDS.csv")
    #df = feather.read_feather('../../data/ben/aug/june18_TCGA.NMJ123.log.RDS.csv.feather')
    # backup = df.copy()
    print("dataset loaded.. preprocessing...")

    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(index='X')

    df = df.reset_index()
    df = df.rename(columns={'index':'Y'})

    df['Y'] = df['Y'].apply(lambda x: x.split(".")[0])
    df['Y'].value_counts().plot(kind='barh', figsize = (10,6))

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
model.add(Dense(1024, activation="relu", input_dim = X_train[:,:-1].shape[1]))
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
    epochs = 2
    history = model.fit(X_train[:,:-1], y_train, 
                        validation_data=(X_test[:,:-1], y_test), 
                        verbose=1, 
                        epochs=epochs, 
                        shuffle=True,
                        callbacks = [ es_callback, cp_callback])
else:
    # The model weights (that are considered the best) are loaded into the model.
    print("Loading model weights...")
    model.load_weights(checkpoint_path)
# break

# # prediction
# preds = model.predict(X_test[:,:-1])
# print("Predicted class: \n", label_encoder.inverse_transform(np.argmax(preds, axis=1)))
# print("Predicted max prob: \n", list((np.max(preds, axis=1)*10000).astype(int)/100))

#TODO
# []

"""## Evaluating with Validation set"""
# Prediction class ID for test set

X_brain_test = X_test[np.argmax(y_test, axis=1) == np.ones_like(np.argmax(y_test, axis=1)) * list(label_encoder.classes_).index(str(args.class_name))][:n_tcga_test]  # Brain has index 1 so ones_like
preds = model.predict(X_brain_test[:,:-1])
preds_brain = np.argmax(preds, axis=1)

# Extract class ID for ground truth
valid_labels = np.ones_like(preds_brain) * list(label_encoder.classes_).index(str(args.class_name))
# This maps groundtruth class encoded  values to class name
real = label_encoder.inverse_transform(valid_labels)  # all brain

# This maps predicted class encoded  values to class name
predicts = label_encoder.inverse_transform(preds_brain)

# dict of real vs prediction classes // for purpose of comparision
# print("real:preds\n",{real[i]:predicts[i] for i in n_tcga_test_range}) 

#TODO uncomment this line to get the labels of TCGA
print("True TCGA labels: \n", real)

print("Predicted tcga class: \n", predicts)
pred_max_prob = list((np.max(preds, axis=1)*10000).astype(int)/100)
print("prediction softmax prob:\n", pred_max_prob)
  # exits after training.

if not args.explain:
    print("skiping SHAP Explaination")
    exit()
else:  
    ## SHAP Explainer
    start = time.time()
    print("Building explainer...")
    def f(X):
        return model.predict([X[:,i] for i in range(X.shape[1])]).flatten()
    
    explainer = shap.KernelExplainer(f, df.iloc[:50,:])
    
    
    print("Elapsed time:", time.time() - start)
    
    X_brain_train = X_train[np.argmax(y_train, axis=1) == np.ones_like(np.argmax(y_train, axis=1))* list(label_encoder.classes_).index(str(args.class_name)) ][:n_tcga_test]  # Brain has index 1 so ones_like

    for set, X_brain in zip(['test', 'train'], [X_brain_test, X_brain_train]):
        if int(args.num_instances) == -1:
            n_tcga_test = X_brain.shape[0]
        for i in range(n_tcga_test):
            fname= f"out/{str(args.class_name)}_{set}set_explain_{i}"
            start = time.time()
            shap_values = explainer.shap_values(X_brain[:,:-1][i], nsamples=500)
            #import pdb; pdb.set_trace()

            if args.save_format == 'pdf':
                shap.force_plot(explainer.expected_value, shap_values, X_test, show=False)
                plt.savefig(f"{fname}_force_plot.pdf")
                plt.clf()
                plt.cla()
                plt.close()
                shap.summary_plot(shap_values, X_brain[:,:-1][i], show=False)
                plt.savefig(f"{fname}_summary_plot.pdf")
                plt.clf()
                plt.cla()
                plt.close()
            if args.save_format == 'json': 
                break
                # out = {}
                # for _i in exp.available_labels():
                #     if label_encoder.classes_[_i] == str(args.class_name):
                #         out[label_encoder.classes_[_i]+'.'+str(X_brain[:,-1][i].astype(int).astype(str))] = exp.as_list(label=_i)
                # with open(fname+'.json', 'w') as f: 
                #     f.write(str(out))

            print(f"{set} {i} Iteration Elapsed time:", time.time() - start)
        print("")
        
    print("All Iteration Elapsed time:", time.time() - start)