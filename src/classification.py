# coding: utf-8

import os,sys

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
import argparse

np.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser(description='arguments for sample number and run number')

parser.add_argument('--sample_srt', type=int, choices=list(range(15)), default=0)
parser.add_argument('--sample_end', type=int, choices=list(range(16)), default=15)
args = parser.parse_args()

from src.data import get_data
from src.models import get_model
from src.train.trainer import train
from src.train.test import predict
from src.explainer import explain

#### Model Definition

data = get_data(load_existing=True, fpkm=True)
(X_train, X_test, y_train, y_test, feature_names, label_encoder, fpkm_data) = data
# import pdb; pdb.set_trace()

input_dim, output_dim = X_train.shape[1], y_train.shape[1]
print("input_dim, output_dim:", input_dim, output_dim)
model = get_model(input_dim, output_dim)

model = train(model, data, load_existing = True)

# predict(model, label_encoder, data)

explain(model, data, explain_data=fpkm_data[args.sample_srt:args.sample_end], srt_idx=args.sample_srt,  n_samples=100, submodular_pick=False)
