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
#Dataframe for training 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# Preprocessing the dataset

df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_COMPLETE_3_Labels.csv')

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



X = preprocess_images(image_paths) 
y = np.array(df.drop(['NAME','Path'],axis=1))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=760, test_size=0.15)
#%%
#Loading the model
model = load_model('../SCRIPTS/TDL/PHYCUV/MODELS/MBNET KFOLD TEST/MobileNet_REG_L2_Lr_00001_fold_0.h5') #CHANGE PATH TO LOAD DESIRED MODEL

# Make predictions on the test data
y_pred = model.predict(X)

# Compute the evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from tensorflow.keras.metrics import Precision, Recall, AUC

from sklearn.metrics import average_precision_score
test_accuracy = accuracy_score(y, y_pred.round())
test_precision = precision_score(y, y_pred.round(), average='samples')
test_recall = recall_score(y, y_pred.round(), average='samples')
test_f1_score = f1_score(y, y_pred.round(), average='samples')
test_hamming_loss = hamming_loss(y, y_pred.round())

# Print the evaluation metrics
print(f'Test accuracy: {test_accuracy}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test f1 score: {test_f1_score}')
print(f'Test hamming loss: {test_hamming_loss}')


print("ANOTHER METRICS")
test_f1_score = f1_score(y, y_pred > 0.5, average=None)
test_precision = Precision()(y, y_pred).numpy()
test_recall = Recall()(y, y_pred).numpy()
test_roc_auc = AUC(curve='ROC')(y, y_pred).numpy()
test_pr_auc = average_precision_score(y, y_pred, average='micro')
print(f'Test F1 score: {test_f1_score}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test ROC AUC: {test_roc_auc}')
print(f'Test PR AUC: {test_pr_auc}')
# %%
#NEWWW

# %%
