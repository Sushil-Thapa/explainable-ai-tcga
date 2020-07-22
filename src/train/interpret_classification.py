# coding: utf-8

import os,sys

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
import random


# ### IMPORTANT
# 
# Change the `nrows` below to train on full dataset.

data_path = "../../data/exT.csv"
# df = pd.read_csv("../data/exT.csv",low_memory=False, nrows=2000)
# df = pd.read_csv(data_path, skiprows=lambda i: i>0 and random.random() > 0.5)
df = pd.read_csv(data_path)


print(df.shape)
df.head()


# df.info(verbose = False)


import pdb; pdb.set_trace()
df = df.rename(columns={'Unnamed: 0':'labels'})
df.head()



# df.describe() # not so helpful because of 19k features



# df.isna().sum().sum()/ len(df) * 100 # check for NaNs, If any..



df['Y'] = df['labels'].apply(lambda x: x.split(".")[0])
df['Y'].head()



df['Y'].value_counts().plot(kind='barh', figsize = (10,6))


# Looks like the dataset is extremely imbalanced. Might have to do do something for it. 

# I tried to visualize these values as well.
# Figure below shows the boxplots for two of these features.


# import seaborn as sns
# sns.boxplot(df['TIGAR'], whis= 3)
# plt.xlim(0, 250)



# import seaborn as sns
# sns.boxplot(df['RAB4B'], whis= 3)
# plt.xlim(0, 250)


# There are some outliers, but not sure we need to remove them for classifier to improve its robustness or use just as it is since this is Gene's data. 
# Also since we are more concerned on the determination of classification, I'd say we can get away with this since it performed satisfactorily.

# ### Label Encoding the categorical variable Y




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()

df['Y'] = label_encoder.fit_transform(df['Y'])
df['Y'].head()


# In[76]:


print(len(label_encoder.classes_))
label_encoder.classes_


# In[77]:


labels = df.pop("labels")
Y = df.pop("Y")


# In[78]:


Y.head()


# ### Converting Y into 1-hot

# In[79]:


Y = keras.utils.to_categorical(Y)  # verify this is of n length not 2
Y


# In[80]:


df.head() # X


# ### Train-Test Split (Stratified and shuffled)

# In[81]:


X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size = 0.2, random_state = 42, stratify=Y, shuffle=True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[85]:


label_encoder.transform(["Skin"])  # Getting the index of Skin //  Just for testing 


# In[88]:


np.argmax(y_test, axis=1)  # Get the indexes from one-hot encoded labels. // Just for testing


# In[92]:


# Hence taking the subset of skin only from testset
X_test_skin = X_test[np.argmax(y_test, axis=1) == 26]
X_test_skin.shape


# You can then use X_test_skin instead of X_test in the explainer for loop.

# Faced MemoryError while standardizing this so doing these steps  
# - reset df variable from memory
# - save backup for splitted data

# In[ ]:


# Variables in memory that's hogging the memories greater than 1MB.
# local_vars = list(locals().items())
# for var, obj in local_vars:
#     size = sys.getsizeof(obj)/1000
#     if size > 1024:
#         print(var, size/1024,"MB")

# df 2439707.544 KB # Evenif the memory was just 2.4GB, it was giving issues occassionally.
# labels 1075.379 KB
# X_train 1951868.152 KB
# X_test 487967.056 KB
# y_train 1737.104 KB


# In[ ]:


# %reset_selective -f "^df$"  # Releasing memory of df variable


# ### Feature Scaling 

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 
X_test.shape


# In[ ]:


filename = "data/scaled_splitted_data.npz"
# filename = "../data/splitted_data_X_train.npz"


# In[ ]:


np.savez_compressed(filename, X_train=X_train, X_test=X_test, 
                    y_train=y_train, y_test=y_test,
                    feature_names=df.columns,
                    class_names=label_encoder.classes_
                   )
print(f"{filename} data saved..")


# Load the splitted variables for next step.
# Just Run this below to load them and start experimentation later.

# In[ ]:


# a = np.load(filename)


# In[ ]:


# X_train, X_test, y_train, y_test = a["X_train"], a["X_test"], a["y_train"], a["y_test"]
# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Model Definition

# In[ ]:


tf.keras.backend.clear_session() # reset keras session


# In[ ]:

def get_model():
    model = Sequential()
    model.add(Dense(1024, activation="relu", input_dim = X_train.shape[1]))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    return model

model = get_model()
model.summary()


# ## Training

# In[ ]:


checkpoint_path = "out/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                  save_best_only=True,
                                                  monitor='val_accuracy',
                                                 verbose=1)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)


# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['categorical_crossentropy','accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    verbose=1, 
                    epochs=20, 
                    shuffle=True,
                    callbacks = [ es_callback, cp_callback])


# ## Evaluating with Validation set

# In[ ]:


predicted_valid_labels = np.argmax(model.predict(X_test), axis=1)
valid_labels = np.argmax(y_test, axis=1)

test_range = range(10)
print("Predicted labels: ", predicted_valid_labels[test_range])
print("True labels: ", valid_labels[test_range])

real = label_encoder.inverse_transform(valid_labels[test_range])
predicts = label_encoder.inverse_transform(predicted_valid_labels[test_range])
print("real:preds\n",{real[i]:predicts[i] for i in test_range})


# In[ ]:


# Visualization of Confusion Matrix
# import seaborn as sns

# cm = metrics.confusion_matrix(valid_labels, predicted_valid_labels)
# # print(cm)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plt.figure(figsize=(20,20))
# sns.heatmap(cm_normalized, annot=True, fmt=".4f", linewidths=.5, square = True, cmap = 'summer')
# plt.xlabel('Predicted Values', size=20)
# plt.ylabel('Actual Values', size=20)

# ticks = np.arange(len(set(valid_labels)))
# tick_marks = ['Adrenal', 'Bile', 'Bladder', 'Bone', 'Brain', 'Breast', 'Cervix',
#        'Colon', 'Esophagus', 'Fallopian', 'Head', 'Kidney', 'Leukemia',
#        'Liver', 'Lung', 'Lymph', 'Mediastinum', 'Nervous', 'Ocular',
#        'Ovarian', 'Pancreas', 'Pelvis', 'Peritoneum', 'Pleura',
#        'Prostate', 'Rectum', 'Sarcoma', 'Skin', 'Stomach', 'Testis',
#        'Thymus', 'Thyroid', 'Uterus', 'none']

# plt.xticks(ticks+0.5 ,tick_marks, rotation=90, size=12) #add 0.5 to ticks to position it at center
# plt.yticks(ticks+0.5 ,tick_marks, rotation=0, size=12)
# # all_sample_title = 'Accuracy Score: {:.4f}'.format(93) # hardcoded this from training logs for now :D
# # plt.title(all_sample_title, size = 30)
# plt.show()


# ## Lime Explainer

# In[ ]:


import lime
import lime.lime_tabular
import time


# we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles

# In[ ]:


# %%prun
start = time.time()
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                   feature_names=df.columns, 
                                                   class_names=label_encoder.classes_)
print("Explainer Elapsed time:", time.time() - start)


# In[ ]:



for _ in range(2):
    start = time.time()
    ith = np.random.randint(0, X_test.shape[0])
    print(ith,"th test sample")
    exp = explainer.explain_instance(X_test[ith], model.predict_proba, num_features=20, top_labels=5)
    
    # exp.show_in_notebook(show_table=True, show_all=True)
    
    for i in exp.available_labels():
        print(i,"/len(exp.available_labels())th class: ", label_encoder.classes_[i])
    #     display(pd.DataFrame(exp.as_list(label=i)))
        display(exp.as_list(label=i))

    print("Iteration Elapsed time:", time.time() - start)


# For this multi-class classification problem, we set the top_labels parameter, so that we only explain the top class with the highest level of probability.

# In[ ]:


# # %%prun
# start = time.time()
# i = np.random.randint(0, X_test.shape[0])
# print(i)
# exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=10, top_labels=5)
# print(time.time() - start)


# In[ ]:


# exp.show_in_notebook(show_table=True, show_all=False)


# feature1 ≤ X means when this feature’s value satisfy this criteria it support class 0.   
# Float point number on the horizontal bars represent the relative importance of these features.

# In[ ]:


# exp.show_in_notebook(show_table=True, show_all=True)


# In[ ]:


# #for easier analysis and further processing
# for i in exp.available_labels():
#     print(label_encoder.classes_[i])
# #     display(pd.DataFrame(exp.as_list(label=i)))
#     display(exp.as_list(label=i))


# In[ ]:




