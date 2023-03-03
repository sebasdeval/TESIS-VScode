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




import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import keras
#from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.preprocessing import image
from keras.layers import BatchNormalization
import PIL as image_lib
from keras.layers.core import Dense
# import keras.utils as image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


#Dataframe for training 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

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


# %%
#Converting the training and test vectors to tensor format for future models that needed
y_tensor_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_tensor_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
X_tensor_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_tensor_test = tf.convert_to_tensor(X_test, dtype=tf.float32)


#__________ MODEL 1________________________________________________________________
#                    VGG16
#%%
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
import tensorflow as tf
from keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.models import load_model, save_model


# Load the pre-trained VGG16 model and remove the last layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = keras.layers.Flatten()(base_model.output)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(7, activation='sigmoid')(x)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Set up the optimizer, loss function, and evaluation metric
optimizer = Adam(learning_rate=1e-4)
loss = 'binary_crossentropy'
metric = AUC()

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=[BinaryAccuracy(), AUC()])

# Train the model

early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/VGG16/my_modelV2.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop,checkpoint])

#%%%



# %%
#________________________________________CODE FOR TESTING IMAGES OVER TRAINED MODEL_________________
from tensorflow.keras.preprocessing import image
img = image.load_img('../SCRIPTS/TDL/PHYCUV/AUSPEC/INCT41_20201114_234500/INCT41_20201114_234500_2.png', target_size=(SIZE,SIZE,3))

img = image.img_to_array(img)
img = img/255.
plt.imshow(img)
img = np.expand_dims(img, axis=0)

classes = np.array(df.columns[2:]) #Get array of all classes
proba = model.predict(img)  #Get probabilities for each class
sorted_categories = np.argsort(proba[0])[:-8:-1]  #Get class names for top 10 categories

#Print classes and corresponding probabilities
for i in range(7):
    print("{}".format(classes[sorted_categories[i]])+" ({:.3})".format(proba[0][sorted_categories[i]]))
#_____________________________________________________________________________________________________________









#########################_________________________________MODEL 2_______________________________#########################

#%%



import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score



# Load the pre-trained ResNet50 model
base_model = tf.keras.applications.ResNet50(
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
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/RESNET50/my_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model for 100 epochs with batch size 32
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# %%
# ____________________TESTING WITH A SMALL DATASET NOT USED IN TRAINING OR TESTING________________________________________________________________


df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_personal.csv')
# Preprocess images creating caracteristic array
X = preprocess_images(image_paths) 
 
# Images path
image_directory ='../SCRIPTS/TDL/PHYCUV/AUSPEC'
#Creating auxiliar Dataframe
df_T = df
# Obtaining labels array in Numpy format
y = np.array(df.drop(['NAME','Path'],axis=1))

#%%
test_loss, test_accuracy = model.evaluate(X, y)

# Print the test loss and accuracy
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
#%%
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

#______________________________________________________END________________________________________________________________
#%%











#____________________5________________
#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from IPython.display import Image
import matplotlib.pyplot as plt
# %%

image = tf.keras.preprocessing.image.load_img('../SCRIPTS/TDL/PHYCUV/AUSPEC/INCT41_20200126_200000/INCT41_20200126_200000_1.png',target_size=(350,350,3),grayscale=True)
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = tf.keras.applications.mobilenet.preprocess_input(input_arr)
plt.imshow(input_arr)
#%%
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)




#______________5_______________________

# %%
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import tensorflow as tf
from keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.models import load_model, save_model

# Load pre-trained MobileNet model with imagenet weights and exclude the top layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#%%
# Add layers on top of the pre-trained model

x = keras.layers.Flatten()(base_model.output)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(7, activation='sigmoid')(x)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers so that they are not trainable
for layer in base_model.layers:
    layer.trainable = False

model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Set up the optimizer, loss function, and evaluation metric
optimizer = Adam(learning_rate=1e-4)
loss = 'binary_crossentropy'
metric = AUC()

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=[BinaryAccuracy(), AUC()])

# Train the model

early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/MobileNet/my_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
history=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop,checkpoint])

#%%

######### MOBILE NET 2########################
#%%import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score

# Load the pre-trained MobileNet model
base_model = MobileNet(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom output layer for multilabel classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
output = Dense(7, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_f1_score', patience=5, verbose=1, mode='max', restore_best_weights=True)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/MobileNet/my_modelV2.h5', monitor='val_f1_score', save_best_only=True, mode='max', verbose=1)

# Train the model for 100 epochs with batch size 32
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# Evaluate the model on the test set using F1 score
y_pred = model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred > 0.5, average='micro')
print(f'Test F1 score: {test_f1_score}')

#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


# Define the hyperparameters to search through
hyperparams = {
    'lr': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150]
}

# Define the pre-trained model to use
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
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(curve='ROC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

# Define the model checkpoint callback to save each trained model
def save_model(epoch, logs):
    model.save(f'../SCRIPTS/TDL/PHYCUV/MODELS/MobileNet/model_{logs["lr"]}_{logs["batch_size"]}_{logs["epochs"]}_{epoch}.h5')

checkpoint = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)

# Use GridSearchCV to find the best combination of hyperparameters
grid = GridSearchCV(
    estimator=model,
    param_grid=hyperparams,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)

# Train the model using GridSearchCV
history = grid.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop, checkpoint])

# Print the best hyperparameters and their scores
print(f'Best hyperparameters: {grid.best_params_}')
print(f'Best score: {grid.best_score_}')

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import time

# Define a function to create the Keras model
def create_model(lr=0.0001):
    start_time = time.time()
    # Load the pre-trained MobileNet model
    base_model = tf.keras.applications.DenseNet121(
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
    'lr':[0.001],
    'batch_size':[16],
    'epochs': [2]
}

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/DenseNet121/New_model_V1.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Set up the GridSearchCV object
grid = GridSearchCV(
    estimator=estimator,
    param_grid=hyperparams,
    scoring='f1_micro',
    n_jobs=-1,
    cv=2,
    #callbacks=[early_stop, checkpoint]
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


# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# Define a function to create the Keras model
def create_model(lr=0.001):
    # Load the pre-trained ResNet50 model
    base_model = tf.keras.applications.ResNet50(
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

    return model

# Wrap the Keras model inside a scikit-learn estimator
estimator = KerasClassifier(model=create_model, lr=0.0001)


# Define the hyperparameter grid to search over
hyperparams = {
    'lr':[0.0001],
    'batch_size':[32],
    'epochs': [1]
}

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('../SCRIPTS/TDL/PHYCUV/MODELS/MobileNet/model3_V5.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Set up the GridSearchCV object
grid = GridSearchCV(
    estimator=estimator,
    param_grid=hyperparams,
    scoring='f1_micro',
    n_jobs=-1,
    cv=2
)

# Train the model using GridSearchCV
history = grid.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop, checkpoint])

# Print the best hyperparameters and their scores
print(f'Best hyperparameters: {grid.best_params_}')
print(f'Best score: {grid.best_score_}')
# %%
