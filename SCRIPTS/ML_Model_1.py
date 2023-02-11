#%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import keras
from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
import keras.utils as image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
# Load the csv into a pandas dataframe
#df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/')


#%%

import numpy as np
import pandas as pd
# from keras.preprocessing import image
import keras.utils as image

def preprocess_images(paths, target_size=(350,350,3)):
    X = []
    for path in paths:
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        X.append(img_array)
    return np.array(X)

img=resnet50.preprocess_input(X)

# Load dataframe
# merged_df = pd.read_csv("path/to/merged_df.csv")

# Get image paths from the 'Path' column
image_paths = merged_df['Path'].values

# Preprocess images
X = preprocess_images(image_paths)   

#%%
Y = merged_df.drop(['Path', 'NAME'],axis = 1)
Y = Y.to_numpy()
#%%
#Y_tens = tf.convert_to_tensor(Y, dtype=tf.int64)
X_tens =  tf.convert_to_tensor(X, dtype=tf.int64)
#%%
img=resnet50.preprocess_input(X)
#%%

x_train, x_test, y_train, y_test = train_test_split(img, Y, test_size=0.15)

import tensorflow as tf
#Considering y variable holds numpy array
#%%
x_train =  tf.convert_to_tensor(x_train, dtype=tf.float64)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float64)
y_test = tf.convert_to_tensor(y_test,dtype=tf.int64)
y_train = tf.convert_to_tensor(y_train,dtype=tf.int64)
#%%
from tensorflow.keras.applications import resnet50

#%%
model_resnet=resnet50.ResNet50(weights='imagenet')

model_resnet.summary()
# display the summary to see the properties of the model
#%%

print("Summary of Custom ResNet50 model.\n")
print("1) We setup input layer and 2) We removed top (last) layer. \n")

# let us prepare our input_layer to pass our image size. default is (224,224,3). we will change it to (100,100,3)
input_layer=layers.Input(shape=(350,350,3))

# initialize the transfer model ResNet50 with appropriate properties per our need.
# we are passing paramers as following
# 1) weights='imagenet' - Using this we are carring weights as of original weights.
# 2) input_tensor to pass the ResNet50 using input_tensor
# 3) we want to change the last layer so we are not including top layer
resnet_model=resnet50.ResNet50(weights='imagenet',input_tensor=input_layer,include_top=False)

resnet_model.summary()
#%%
# access the current last layer of the model and add flatten and dense after it

print("Summary of Custom ResNet50 model.\n")
print("1) We flatten the last layer and added 1 Dense layer and 1 output layer.\n")

last_layer=resnet_model.output # we are taking last layer of the model

# Add flatten layer: we are extending Neural Network by adding flattn layer
flatten=layers.Flatten()(last_layer) 

# Add dense layer
# dense1=layers.Dense(100,activation='relu')(flatten)

# Add dense layer to the final output layer
output_layer=layers.Dense(6,activation='softmax')(flatten)

# Creating modle with input and output layer
model=models.Model(inputs=input_layer,outputs=output_layer)

# Summarize the model
model.summary()

#%%
# we will freez all the layers except the last layer

# we are making all the layers intrainable except the last layer
print("We are making all the layers intrainable except the last layer. \n")
for layer in resnet_model.layers[:-1]:
    layer.trainable=False
#%%

# Compiling Model

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print("Model compilation completed.")
#%%

# Fit the Model

model.fit(x_train,y_train,epochs=20,batch_size=64,verbose=True,validation_data=(x_test,y_test))

print("Fitting the model completed.")

#%%


img=image.load_img('../SCRIPTS/TDL/PHYCUV/AUSPEC/INCT41_20200217_001500/INCT41_20200217_001500_7.png',target_size=(350,350))

img=image.img_to_array(img)

import matplotlib.pyplot as plt
plt.imshow(img.astype('int32'))
plt.show()

img=resnet50.preprocess_input(img)
prediction=model.predict(img.reshape(1,350,350,3))
output=np.argmax(prediction)
 
 
 #%%
 
 
 def preprocess_images(paths, target_size=(350,350,3)):
    def load_and_preprocess_image(path):
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        return resnet50.preprocess_input(np.array([img_array]))

    images = tf.map_fn(load_and_preprocess_image, paths, dtype=tf.float64)
    return images

# Get image paths and labels
image_paths = merged_df['Path'].values
labels = merged_df.drop(['Path', 'NAME'], axis=1).to_numpy()

# Create dataset from image paths and labels
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(lambda x, y: (preprocess_images(x), y))

# Split the dataset into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(img, Y, test_size=0.15)


#%%
batch_size = 64
train_dataset = dataset.batch(batch_size).shuffle(10000)
test_dataset = x_test.batch(batch_size)

model.fit(train_dataset, epochs=4, verbose=True, validation_data=test_dataset)

#%%
optimizer = tf.keras.optimizers.AdamW()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])