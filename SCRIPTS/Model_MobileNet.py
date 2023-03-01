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

# Define a function to create the Keras model
def create_model(lr=0.0001):
    start_time = time.time()
    # Load the pre-trained MobileNet model
    base_model = tf.keras.applications.MobileNet(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom output layer for multi-label classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = Dense(7, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
    )
    end_time = time.time()
    duration = end_time - start_time
    return model

# Wrap the Keras model inside a scikit-learn estimator
estimator = KerasClassifier(model=create_model, lr=0.0001)


# Define the hyperparameter grid to search over
hyperparams = {
    'lr':[0.0001,0.00001],
    'batch_size':[16,32,64],
    'epochs': [20,30,40,50]
}

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/MobileNet/model_V1.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Set up the GridSearchCV object
grid = GridSearchCV(
    estimator=estimator,
    param_grid=hyperparams,
    scoring='f1_micro',
    n_jobs=-1,
    cv=2
)

# Train the model using GridSearchCV
start_time = time.time()
history = grid.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop, checkpoint])
end_time = time.time()
duration = end_time - start_time
print(f"Training the model took {duration:.2f} seconds")

# Print the best hyperparameters and their scores
print(f'Best hyperparameters: {grid.best_params_}')
print(f'Best score: {grid.best_score_}')
