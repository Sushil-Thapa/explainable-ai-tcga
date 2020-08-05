# coding: utf-8

import os,sys

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random

np.random.seed(42)
random.seed(42)


from src.data import get_data
from src.models import get_model
from src.train.trainer import train
from src.train.test import predict
from src.explainer import explain

#### Model Definition

data = get_data(load_existing=False, fpkm=True)
(X_train, X_test, y_train, y_test, feature_names, label_encoder) = data
# import pdb; pdb.set_trace()

input_dim, output_dim = X_train.shape[1], y_train.shape[1]
print("input_dim, output_dim:", input_dim, output_dim)
model = get_model(input_dim, output_dim)

model = train(model, data, load_existing = True)

# predict(model, label_encoder, data)

explain(model, data, n_samples=1, submodular_pick= False)