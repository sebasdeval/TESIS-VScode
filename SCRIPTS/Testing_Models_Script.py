#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from maad import sound, util
import glob
from os import walk
import sphinx
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import wave
from tensorflow.keras.models import Sequential
import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import PIL as image_lib
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import time
from tensorflow.keras.models import load_model


# Preprocessing the dataset

df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_personal.csv')

#%%
#Function for preprocesing images
def preprocess_images(paths, target_size=(224,224,3)):
    X = []
    for path in paths:
        img = image.load_img(path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array/255     
        X.append(img_array)
    return np.array(X)
image_paths = df['Path'].values

#%%

X = preprocess_images(image_paths) 
y = np.array(df.drop(['NAME','Path'],axis=1))

#Loading the model
model = load_model('../SCRIPTS/TDL/PHYCUV/MODELS/RESNET50/my_model.h5') #CHANGE PATH TO LOAD DESIRED MODEL

# Make predictions on the test data
y_pred = model.predict(X)

# Compute the evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
test_accuracy = accuracy_score(y, y_pred.round())
test_precision = precision_score(y, y_pred.round(), average='micro')
test_recall = recall_score(y, y_pred.round(), average='micro')
test_f1_score = f1_score(y, y_pred.round(), average='micro')
test_hamming_loss = hamming_loss(y, y_pred.round())

# Print the evaluation metrics
print(f'Test accuracy: {test_accuracy}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test f1 score: {test_f1_score}')
print(f'Test hamming loss: {test_hamming_loss}')