""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import numpy as np
import pickle
import pdb
import skimage.data
import skimage.transform
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import backend as K

from imutils import paths


def load_data(trainingPath, testingPath):
    data = []
    labels = []
    test_data = []
    test_labels = []

    # loop over the training images
    for imagePath in paths.list_files(trainingPath, validExts=(".png",".ppm")):
        image = skimage.data.imread(imagePath)
        resized_image = skimage.transform.resize(image, (32, 32, 3))
        labels.append(int(imagePath.split("/")[-2]))
        data.append(resized_image)
    
     # loop over the test images
    for imagePath in paths.list_files(testingPath, validExts=(".png",".ppm")):
        image = skimage.data.imread(imagePath)
        resized_image = skimage.transform.resize(image, (32, 32, 3))
        test_labels.append(int(imagePath.split("/")[-2]))
        test_data.append(resized_image)

    return (data, labels, test_data, test_labels)


train_data_path = "../images/training"
testing_data_path = "../images/testing"

train_images, labels, test_images, test_labels = load_data(train_data_path, testing_data_path)

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)
test_images = np.array(train_images)
test_labels = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels = labels / 62
test_labels = test_labels / 62

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)

# Test size may be 10% or 20%
#X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

X_train = train_images
y_train = labels

X_val = test_images
y_val = test_labels


# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 32
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]
#pdb.set_trace()

### Here is the actual neural network ###
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

# Conv Layer 2
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

# Pooling 1
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 3
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))

# Pooling 2
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 6
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(62))

### End of network ###


# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

# Compiling and training the model
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
model.summary()

y_train = y_train.reshape((-1,1))
y_val = y_val.reshape((-1,1))


gene = datagen.flow(X_train, y_train, batch_size=batch_size)
#pdb.set_trace()

model.fit_generator(gene, steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

# Freeze layers since training is done
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error')

# Save model architecture and weights
model.save('full_CNN_model.h5')