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
import tensorflow as tf


# Preprocessing the dataset

df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df.csv')

#df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_personal.csv')

#%%
# Eliminating all rows that contain no species identificated in the spectrograms
# Find the sum of each row for the last 6 columns
row_sums = df.iloc[:, -6:].sum(axis=1)

# Keep only the rows with a non-zero sum in the last 6 columns
df_no_ze = df[row_sums != 0]
df_all_ze = df[row_sums == 0]

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

# Images path
image_directory ='../SCRIPTS/TDL/PHYCUV/AUSPEC'
#Creating auxiliar Dataframe
df_T = df
# Preprocess images creating caracteristic array
X = preprocess_images(image_paths) 
 # Obtaining labels array in Numpy format
y = np.array(df.drop(['NAME','Path'],axis=1))
#Declaring size of mages
SIZE = 224
#%%
# Dividing Dataset in training and testing with 20 percent of whole dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=760, test_size=0.2)

# %%
from sklearn.utils import resample
import numpy as np

# Convert y to integer labels (assuming one-hot encoding)
y = np.argmax(y, axis=1)

# Count the number of samples in each class
class_counts = np.bincount(y)

# Determine the maximum number of samples in a class
max_class_count = np.max(class_counts)

# Oversample each class to have the same number of samples as the maximum class count
X_oversampled = []
y_oversampled = []
for class_index, class_count in enumerate(class_counts):
    X_class = X[y == class_index]
    y_class = y[y == class_index]
    X_oversampled_class, y_oversampled_class = resample(X_class, y_class, n_samples=max_class_count, replace=True, random_state=42)
    X_oversampled.append(X_oversampled_class)
    y_oversampled.append(y_oversampled_class)

# Concatenate the oversampled data
X_oversampled = np.concatenate(X_oversampled)
y_oversampled = np.concatenate(y_oversampled)

# Convert y back to one-hot encoding
y_oversampled = np.eye(7)[y_oversampled]


# %%
X = X_oversampled
y = y_oversampled 

#%%
shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]