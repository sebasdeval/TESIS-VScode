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
# Dividing Dataset in training and testing with 20 percent of whole dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=760, test_size=0.2)
#X_tensor_train, X_tensor_test, y_tensor_train, y_tensor_test = train_test_split(X_tensor, y_tensor, random_state=20, test_size=0.2)

#%%
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

# Load the pre-trained MobileNet model
base_model = tf.keras.applications.DenseNet121(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom output layer for multilabel classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)
x = keras.layers.Dropout(0.5)(x)
output = Dense(7, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    #metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
    metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/DenseNet121/DenseNet_Reg_l2_lr01.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model for 100 epochs with batch size 32
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose = 1
)

# Evaluate the model on the test set using F1 score
y_pred = model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred > 0.5, average='micro')
test_precision = Precision()(y_test, y_pred).numpy()
test_recall = Recall()(y_test, y_pred).numpy()
test_roc_auc = AUC(curve='ROC')(y_test, y_pred).numpy()
test_pr_auc = average_precision_score(y_test, y_pred, average='micro')
print(f'Test F1 score: {test_f1_score}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test ROC AUC: {test_roc_auc}')
print(f'Test PR AUC: {test_pr_auc}')
#%%

