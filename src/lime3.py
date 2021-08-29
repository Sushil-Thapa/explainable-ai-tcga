import os,sys

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.utils
import matplotlib.pyplot as plt
import seaborn as sns

import lime
import lime.lime_tabular
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn import metrics
import random

epochs = 3
random_state = 42
train_flag = False

# number of test set sample(tcga) for inspection
n_tcga_test = 30
n_tcga_test_range = range(-n_tcga_test,0)

# Load dataset
# tcga_path = "/data/cancer/GDC/new/metadata/LexT.csv"
# tcga_path = "../exT.csv"
tcga_path = "../TCGA.NMJ12.log.csv"
tcga_path = "TCGA.NMJ12.log.csv"
df = pd.read_csv(tcga_path)

# rename labels column
df = df.rename(columns={'Unnamed: 0':'labels'})

# extract last 30 columns  for stem cell
df, stem_df = df.copy().iloc[:-30,:], df.iloc[-30:,:]

# map numbered labels to single name
df['Y'] = df['labels'].apply(lambda x: x.split(".")[0])

#plot the distribution 
# df['Y'].value_counts().plot(kind='barh', figsize = (10,6))

# encode class names with IDs
label_encoder = LabelEncoder()
df['Y'] = label_encoder.fit_transform(df['Y'])
print(f"Number of classes:{len(label_encoder.classes_)}\n\nClasses:\n{label_encoder.classes_}")


_ = df.pop("labels")
Y = df.pop("Y")

"""### Converting Y into 1-hot"""
Y = keras.utils.to_categorical(Y)  

### Prepare stem Cell dataset
stem_day = stem_df.pop('labels') # remove Stem Info
stem_X = stem_df.copy() # Remaining stem cell feats

#TODO to Apply Log on tcga dataset
# This line applies log on all the dataset df // df is only the tcga part of the csv Ben provided
# please uncomment and change the last np.exe(-8) part to any other value as necessary for eg 10e-4
# df = df.astype(float).applymap(lambda x: np.log(x + np.exp(-8)))
# stem_X = stem_X.astype(float).applymap(lambda x: np.log(x + np.exp(-8)))

"""### Train-Test Split (Stratified and shuffled)"""
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size = 0.2, random_state = random_state, stratify=Y, shuffle=True)

#TODO uncomment this to run standaridization
"""### Feature Standardization"""
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test) 

"""### Model Definition& Config"""
tf.keras.backend.clear_session() # reset keras session
model = Sequential()
model.add(Dense(1024, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(y_train.shape[1], activation="softmax"))
model.summary()

## Add model training configs
checkpoint_path = "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                 verbose=1)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['categorical_crossentropy','accuracy'])

if train_flag:  # just load model and get explanations if not training
    """## Training"""
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        verbose=1, 
                        epochs=epochs, 
                        shuffle=True,
                        callbacks = [ es_callback, cp_callback])

    # StemCell prediction
    preds_stem = model.predict(stem_X)
    print("Predicted stem class: \n", label_encoder.inverse_transform(np.argmax(preds_stem, axis=1)))
    print("Predicted stem max prob: \n", list((np.max(preds_stem, axis=1)*10000).astype(int)/100))


    """## Evaluating with Validation set"""
    # Prediction class ID for test set
    preds = model.predict(X_test)
    predicted_valid_labels = np.argmax(preds, axis=1)

    #TODO uncomment this to get real ground truth values from TCGA
    # Extract class ID for ground truth
    valid_labels = np.argmax(y_test, axis=1)

    # This maps groundtruth class encoded  values to class name
    real = label_encoder.inverse_transform(valid_labels[n_tcga_test_range])

    # This maps predicted class encoded  values to class name
    predicts = label_encoder.inverse_transform(predicted_valid_labels[n_tcga_test_range])

    # dict of real vs prediction classes // for purpose of comparision
    # print("real:preds\n",{real[i]:predicts[i] for i in n_tcga_test_range}) 

    #TODO uncomment this line to get the labels of TCGA
    print("True TCGA labels: \n", real)

    print("Predicted tcga class: \n", predicts)
    pred_max_prob = list((np.max(preds, axis=1)[n_tcga_test_range]*10000).astype(int)/100)
    print("prediction tcga max prob:\n", pred_max_prob)
    exit()  # exits after training.

else:
    # The model weights (that are considered the best) are loaded into the model.
    print("Loading model weights...")
    model.load_weights(checkpoint_path)

    ## Lime Explainer
    # we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles
    start = time.time()
    print("Building explainer...")
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=df.columns, class_names=label_encoder.classes_)
    print("Elapsed time:", time.time() - start)

    for i in n_tcga_test_range:
        ith = 31+i
        fname="out_" + str(ith)
        start = time.time()
        exp = explainer.explain_instance(stem_X.to_numpy()[i], model.predict_proba, num_features=20, top_labels=1, num_samples=10000)
        exp.save_to_file(fname,show_table=True)
        print(f"{ith} Iteration Elapsed time:", time.time() - start)
        

    #exp.show_in_notebook(show_table=True, show_all=True)
        
    #for i in exp.available_labels():

    exp.save_to_file("ss.1.html",show_table=True, show_all=True)
    print("All Iteration Elapsed time:", time.time() - start)




"""
# For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.

# # %%prun
# start = time.time()
# i = np.random.randint(0, X_test.shape[0])
# print(i)
# exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=10, top_labels=5)
# print(time.time() - start)

# exp.show_in_notebook(show_table=True, show_all=False)

# eature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
# Float point number on the horizontal bars represent the relative importance of these features.


# exp.show_in_notebook(show_table=True, show_all=True)

# #for easier analysis and further processing
# for i in exp.available_labels():
#     print(label_encoder.classes_[i])
# #     display(pd.DataFrame(exp.as_list(label=i)))
#     display(exp.as_list(label=i))

"""
# lime explainer end