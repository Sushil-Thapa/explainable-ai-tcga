import os,sys
import math
import json

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.utils
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from collections import Counter

import random
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

import torch
from pytorch_tabnet.tab_model import TabNetClassifier

def get_dataset():
    print("loading dataset")
    df = pd.read_csv("/Users/thapasushil/project/lanl/genomics/explainable-ai-tcga/data/ben/aug/june18_TCGA.NMJ123.log.RDS.csv")
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
    # Y = keras.utils.to_categorical(Y)
    print("returninig dataset")
    return df, Y, label_encoder



def main():
    df, Y, label_encoder = get_dataset()
    global_exp_dict = {}
    local_exp_dict = {}

    for random_state in range(2):
        print(f"selecting random state{random_state} to split")
        X_train, X_test, y_train, y_test = train_test_split(df.to_numpy(), Y.to_numpy(), test_size = 0.25, random_state = random_state, stratify=Y, shuffle=True)
        X_valid, y_valid = X_test.copy(), y_test.copy()  # not enough dataset
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        print(f"initializing model")

        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-2),
            scheduler_params = {"gamma": 0.95,
                            "step_size": 5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-4,
            seed = random_state
        )
        print(model)

        print("training model")
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            batch_size=256, virtual_batch_size=32,
        ) 

        #save models
        model_name = f"out/model_seed{random_state}"
        saved_filepath = model.save_model(model_name)
        print(f"saved model to {saved_filepath}")

        # plot loss
        plt.title(f"Validation Loss:{max(model.history['loss']):.3f}")
        plt.xlabel("epochs")
        plt.ylabel("val loss")
        plt.plot(model.history['loss'])
        plt.savefig(f"out/plots/loss_seed{random_state}.png")
        print(f"saved loss plot out/plots/loss_seed{random_state}.png")

        # plot accuracy
        t_acc = max(model.history['train_accuracy'])
        v_acc = max(model.history['valid_accuracy'])
                    
        plt.plot(model.history['train_accuracy'], label='train')
        plt.plot(model.history['valid_accuracy'], label='val')
        plt.title(f"Train/Val Accuracy: {t_acc:.2f}/{v_acc:.2f}")
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper left')
        plt.savefig(f"out/plots/accuracy_seed{random_state}.png")
        print(f"saved acc plot out/plots/accuracy_seed{random_state}.png")


        # global explanations
        imp = model.feature_importances_
        dfc = df.columns

        feats = {dfc[id]: imp[id] for id in range(imp.shape[0]) if imp[id] > 0 }  # get feats with nonzero global attention weights
        sorted_feats = sorted(feats.copy().items(), key=lambda x: x[1], reverse=True) # sort based on values
        global_exp_dict[random_state] = list(dfc[imp.argsort()[-imp[imp>0].shape[0]:][::-1]])

        name2idx = {v:i for i,v in enumerate(label_encoder.classes_)}


        all_data = [('train',X_train, y_train),("test",X_test, y_test)]

        # Local Explanations 
        local_exp_dict[random_state] = {}
        for d_name, X_t, y_t in all_data:
            local_exp_dict[random_state][d_name] = {}
            explain_matrix, masks = model.explain(X_t)

            class_dist = Counter([label_encoder.classes_[i] for i in y_t])

            plt.figure(figsize=(15,15))
            plt.barh(list(class_dist.keys()), class_dist.values())
            plt.title("Test set Distribution")
            for i, v in enumerate(list(class_dist.keys())):
                plt.text(class_dist[v] + 3, i-0.25, class_dist[v], color='blue')
            plt.savefig(f"out/plots/data_distribution_{d_name}_seed{random_state}.png")
            print(f"saved data dist plot data_distribution_{d_name}_seed{random_state}.png")

            for class_name in label_encoder.classes_:
                local_exp_dict[random_state][d_name][class_name] = {}
                flag = y_t==name2idx[class_name] # get flags with class_name
                X_local = X_t[flag] # only select class_name X_test data subset
                explain_local = explain_matrix[flag] # get global explanations for such subset

                local_exp_counter = Counter()
                for i in range(class_dist[class_name]):
                    # local_exp_dict[random_state][d_name][class_name][i] = {}  # optional ith brain sample in train type of local exp
                    imp = explain_local[i]
                    idx = list(dfc[imp.argsort()[-imp[imp>0].shape[0]:][::-1]]) # idx of descending weighted per features 
                    for _idx in idx:
                        local_exp_counter.update({dfc[_idx]:1})

                plt.figure(figsize=(30,30))
                local_exp_counter = dict(local_exp_counter.most_common())

                local_exp_dict[random_state][d_name][class_name] = local_exp_counter
                plt.barh(list(local_exp_counter.keys()), local_exp_counter.values())

                plt.title(f"Cumulative Local Explanations: all {d_name}_seed{random_state}_class{class_name} samples")
                for i, v in enumerate(list(local_exp_counter.keys())):
                    plt.text(local_exp_counter[v] + 3, i-0.25, local_exp_counter[v], color='blue')
                plt.savefig(f"out/plots/local_exp_{d_name}_seed{random_state}_class{class_name}.png")
                print(f"saved local exp plot local_exp_{d_name}_seed{random_state}_class{class_name}.png")

    # save global explanations
    dump_json_path = 'out/xai/global_exp.json'
    with open(dump_json_path, 'w') as fp:
        json.dump(global_exp_dict, fp)
    
    dump_json_path = 'out/xai/local_exp.json'
    with open(dump_json_path, 'w') as fp:
        json.dump(local_exp_dict, fp)

if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
