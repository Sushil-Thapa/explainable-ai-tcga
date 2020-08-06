import os,sys

import numpy as np
import pandas as pd
import random

import keras.utils

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import pickle


def get_data(load_existing=True, fpkm=False):
    
    filename = "data/scaled_splitted_data.pickle"
    data_path = "../../data/exT.csv"
    if fpkm:
        filename = "data/scaled_splitted_data_with_fpkm.pickle"

    if load_existing == False:
        # ### IMPORTANT
        # 
        # Change the `nrows` below to train on full dataset.
        print(f"Loading dataset:{data_path}")

        # df = pd.read_csv(data_path,low_memory=False, nrows=1000) # First 2k dataset rows
        # df = pd.read_csv(data_path, skiprows=lambda i: i>0 and random.random() > 0.8)  # random 50% of dataset rows

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


        if fpkm == True:
            
            fpkm_path = "../../data/FPKM_gene_counts_FPKM.csv"

            print(f"Loading dataset:{fpkm_path}")

            fpkm_df = pd.read_csv(fpkm_path)
            print(f"Loaded dataset with shape:{fpkm_df.shape}")
            import pdb; pdb.set_trace()
            fpkm_df = fpkm_df.dropna(axis=1)

            fpkm_df = fpkm_df.applymap(lambda x: str(x).strip('gene-'))
            fpkm_df = fpkm_df.set_index("Unnamed: 0").T

            fpkm_df = fpkm_df.rename_axis(None).reset_index(drop=True)
            fpkm_df = fpkm_df.astype(float).applymap(lambda x: np.log2(x + 1))


            intersect_cols = np.intersect1d(df.columns, fpkm_df.columns)

            print("Extracting interseted dataset.")
            df = df[intersect_cols]
            fpkm_df = fpkm_df[intersect_cols]
        # else:
            fpkm_data = fpkm_df.to_numpy()
        

        # ### Train-Test Split (Stratified and shuffled)
        X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size = 0.2, random_state = 42, stratify=Y, shuffle=True)
        feature_names = df.columns
        
        fpkm_df = None
        df = None

        # ### Feature Scaling 
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 

        if fpkm == True:
            fpkm_data = scaler.transform(fpkm_data)
        else:
            fpkm_data = None



        if os.path.isfile(filename):
            print("File Exists...")
            sys.exit()

        # np.savez_compressed(filename, X_train=X_train, X_test=X_test, 
        #                     y_train=y_train, y_test=y_test,
        #                     feature_names=df.columns,
        #                     label_encoder=label_encoder,
        #                 )

        temp = {
            "X_train":X_train, "X_test":X_test, 
            "y_train":y_train, "y_test":y_test,
            "feature_names":feature_names,
            "label_encoder":label_encoder,

            "fpkm_data" : fpkm_data
        }

        with open(filename, 'wb') as f:
            pickle.dump(temp, f)

        print(f"{filename} data saved..")

    else:
        # Load the splitted variables for next step.
        # Just Run this below to load them and start experimentation later.

        # a = np.load(filename)
        with open(filename, 'rb') as f:
            a = pickle.load(f)

        X_train, X_test, y_train, y_test = a["X_train"], a["X_test"], a["y_train"], a["y_test"]
        feature_names, label_encoder = a["feature_names"], a["label_encoder"]
        print(f"Successfully loaded from existing dump {filename}")

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return (X_train, X_test, y_train, y_test, feature_names, label_encoder)

