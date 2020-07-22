import os

import numpy as np
import pandas as pd

import keras.utils

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import pickle

# a = {'hello': 'world'}

# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)


def get_data(load_existing=True):
    
    filename = "data/scaled_splitted_data.npz"
    data_path = "../../data/exT.csv"

    if load_existing == False:
        # ### IMPORTANT
        # 
        # Change the `nrows` below to train on full dataset.

        
        # df = pd.read_csv("../data/exT.csv",low_memory=False, nrows=2000) # First 2k dataset rows
        # df = pd.read_csv(data_path, skiprows=lambda i: i>0 and random.random() > 0.5)  # random 50% of dataset rows
        print(f"Loading dataset:{data_path}")

        df = pd.read_csv(data_path)
        print(f"Loaded dataset with shape:{df.shape}")

        # df.info(verbose = False)

        df = df.rename(columns={'Unnamed: 0':'labels'})

        # df.describe() # not so helpful because of 19k features
        # df.isna().sum().sum()/ len(df) * 100 # check for NaNs, If any..

        df['Y'] = df['labels'].apply(lambda x: x.split(".")[0])
        df['Y'].value_counts().plot(kind='barh', figsize = (10,6))

        # ### Label Encoding the categorical variable Y
        
        label_encoder = LabelEncoder()

        df['Y'] = label_encoder.fit_transform(df['Y'])
        print(f"{len(label_encoder.classes_)}: {label_encoder.classes_}")
        

        labels = df.pop("labels")
        Y = df.pop("Y")

        # ### Converting Y into 1-hot

        Y = keras.utils.to_categorical(Y)  # verify this is of n length not 2

        # ### Train-Test Split (Stratified and shuffled)
        X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size = 0.2, random_state = 42, stratify=Y, shuffle=True)
        

        # ### Feature Scaling 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 

        feature_names = df.columns

        if os.path.isfile(filename):
            print("File Exists...")
            return 

        np.savez_compressed(filename, X_train=X_train, X_test=X_test, 
                            y_train=y_train, y_test=y_test,
                            feature_names=df.columns,
                            label_encoder=label_encoder,
                        )
        print(f"{filename} data saved..")

    else:
        # Load the splitted variables for next step.
        # Just Run this below to load them and start experimentation later.

        a = np.load(filename)

        X_train, X_test, y_train, y_test = a["X_train"], a["X_test"], a["y_train"], a["y_test"]
        feature_names, label_encoder = a["feature_names"], a["label_encoder"]
        print(f"Successfully loaded from existing dump {filename}")

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return (X_train, X_test, y_train, y_test, feature_names, label_encoder)

