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


# # Preprocessing the dataset

# df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_COMPLETE_3_Labels.csv',delimiter=',')

# #df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_personal.csv')


# # Eliminating all rows that contain no species identificated in the spectrograms
# # Find the sum of each row for the last 6 columns
# row_sums = df.iloc[:, -6:].sum(axis=1)

# # Keep only the rows with a non-zero sum in the last 6 columns
# df_no_ze = df[row_sums != 0]
# df_all_ze = df[row_sums == 0]

# #Function for preprocesing images
# def preprocess_images(paths, target_size=(224,224,3)):
#     X = []
#     for path in paths:
#         img = image.load_img(path, target_size=target_size)
#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = img_array/255     
#         X.append(img_array)
#     return np.array(X)
# image_paths = df['Path'].values


# # Images path
# image_directory ='../SCRIPTS/TDL/PHYCUV/AUSPEC'
# #Creating auxiliar Dataframe
# df_T = df
# # Preprocess images creating caracteristic array
# X = preprocess_images(image_paths) 
#  # Obtaining labels array in Numpy format
# y = np.array(df.drop(['NAME','Path'],axis=1))
# #Declaring size of mages
# SIZE = 224
# # Dividing Dataset in training and testing with 20 percent of whole dataset for testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # verify the distribution of labels in the train and test sets
# import numpy as np
# train_label_counts = np.sum(y_train, axis=0)
# test_label_counts = np.sum(y_test, axis=0)
# print(f"Train label counts: {train_label_counts}")
# print(f"Test label counts: {test_label_counts}")
#%%



from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import average_precision_score

# Load the pre-trained MobileNet model
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom output layer for multilabel classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)
x = keras.layers.Dropout(0.5)(x)
output = Dense(3, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    #metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
    metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True,start_from_epoch=6)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/NEW_MODELS/Not_Augmented/One_Fully_Connected_Layers/Regularization L2/Resnet50/DenseNet121_1LYR_RegL2_Lr_00001.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

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
test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
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
#2LAYERS

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import average_precision_score

# Load the pre-trained MobileNet model
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom output layer for multilabel classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)
x = keras.layers.Dropout(0.5)(x)
x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)
x = keras.layers.Dropout(0.5)(x)
output = Dense(3, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    #metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
    metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True,start_from_epoch=6)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/NEW_MODELS/Not_Augmented/Two_Fully_Connected_Layers/Regularization L2/Resnet50/DenseNet121_2LYR_RegL2_Lr_00001.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

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
test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
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











#K-FOLD VERSION
#1LAYER
# import necessary libraries
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

# Define the number of folds
n_splits = 5

# Initialize the KFold object
kf = KFold(n_splits=n_splits)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Load the pre-trained MobileNet model
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom output layer for multilabel classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    x = keras.layers.Dropout(0.5)(x)
    #x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    #x = keras.layers.Dropout(0.5)(x)
    output = Dense(3, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
    )

    # Set up early stopping and model checkpoint callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True,start_from_epoch=6)
    checkpoint = ModelCheckpoint(f'ResNet50_REG_L2_1LYR_Lr_00001_fold_{fold}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the model for 100 epochs with batch size 32
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

# Evaluate the model on the test set using F1 score
y_pred = model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
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




#2LAYER
# import necessary libraries
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

# Define the number of folds
n_splits = 5

# Initialize the KFold object
kf = KFold(n_splits=n_splits)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Load the pre-trained MobileNet model
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom output layer for multilabel classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    x = keras.layers.Dropout(0.5)(x)
    output = Dense(3, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
    )

    # Set up early stopping and model checkpoint callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True,start_from_epoch=6)
    checkpoint = ModelCheckpoint(f'MobileNet_REG_L2_2LYR_Lr_00001_fold_{fold}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the model for 100 epochs with batch size 32
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

# Evaluate the model on the test set using F1 score
y_pred = model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
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