# USAGE
# python main.py -t ../images/training  -e ../images/testing
# python main.py -t ../ROI_images/training -e ../ROI_images/testing

import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt

from imutils import paths
import argparse
import cv2
import numpy as np
from localbinarypatterns import LocalBinaryPatterns

from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from numpy import array

def load_data(data_dir):
    directories = []
    # parse data directory.
    for d in os.listdir(data_dir):
        # find image directories and store.
        if os.path.isdir(os.path.join(data_dir, d)):
            directories.append(d)
    # empty lists for labels and images.
    labels = []
    images = []
    # parse directories.
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = []
        # parse sub-folders.
        for f in os.listdir(label_dir):
            if f.endswith(".ppm"):
                # find images, store to list.
                file_names.append(os.path.join(label_dir, f))
        for f in file_names:
            # read images and corresponding labels.
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

def check_images(images):
    for image in images:
        print("Shape: {}, Intensity Min: {}, Intensity Max: {}".format(image.shape, image.min(), image.max()))
    print("\n")

def get_LBP_Features(dataSetPath):

    # initialize the local binary patterns descriptor along with the data and label lists
	desc = LocalBinaryPatterns(24, 8)
	data = []
	labels = []

    # loop over the training images
	for imagePath in paths.list_files(dataSetPath, validExts=(".png",".ppm")):
		
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)

		# extract the label from the image path, then update the
		# label and data lists
		labels.append(imagePath.split("/")[-2])
		data.append(hist)

        return labels, data


# pyTorch model.
class signNet(torch.nn.Module):
    
    def __init__(self, d_in, d_out):
        
        super(signNet, self).__init__()
        self.linear = torch.nn.Linear(d_in, d_out)
        self.relu = torch.nn.ReLU()

    def forward(self, input_features): 
        linearOut = self.linear(input_features)
        actvOut = self.relu(linearOut)
        return F.log_softmax(actvOut)

def main(lr, train_data_dir="../ROI_images/training", test_data_dir="../ROI_images/testing"):
    
    # define globals.
    SIZE_W = 32
    SIZE_H = 32
    CHANNELS = 3
    
    d_in = SIZE_W * SIZE_H * CHANNELS
    d_out = 62
    
    learning_rate = lr
    num_of_epochs = 10000
    log_freq = 100
    # get images and labels.
    images, labels = load_data(train_data_dir)
    test_images, test_labels = load_data(test_data_dir)

    check_images(images[:5])
    # print number of train set and number of unique labels.
    print ("There are total {} unique labels in train set for {} images.\n".format(len(set(labels)), len(images)))
    # resize images.
    images_ = []
    test_images_ = []
    for image in images:
        images_.append(skimage.transform.resize(image, (SIZE_W, SIZE_H, CHANNELS), mode='constant'))
    
    for image in test_images:
        test_images_.append(skimage.transform.resize(image, (SIZE_W, SIZE_H, CHANNELS), mode='constant'))
    check_images(images_[:5])
    
    images_array = array(images_)
    labels_array = array(labels)

    graph = tf.Graph()

    with graph.as_default():
        images_ph = tf.placeholder(tf.float32, [None, SIZE_W, SIZE_H, CHANNELS])
        labels_ph = tf.placeholder(tf.int32, [None])

        images_flatten = tf.contrib.layers.flatten(images_ph)

        logits = tf.contrib.layers.fully_connected(images_flatten, d_out, tf.nn.leaky_relu)

        predicted_labels = tf.argmax(logits, 1)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits                                                                                      ,labels = labels_ph))
        train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
    
    session = tf.Session(graph=graph)
    _ = session.run([init])
    
    loss_array = []
    
    # train defined tf model.
    for epoch in range(num_of_epochs):
        _, loss_value = session.run([train, loss], feed_dict = {images_ph: images_array, labels_ph: labels_array})
        if epoch % log_freq == 0:
            print("Loss at instance {}/{}: {}".format(epoch+log_freq, num_of_epochs, loss_value))
            loss_array.append(loss_value)

    #test the model.
    predicted = session.run([predicted_labels], 
                            feed_dict={images_ph: test_images_})[0]
    match_count = sum([int(y==y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = float(match_count) / float(len(test_labels))
    
    return loss_array, accuracy

    
    # create siggNet object.

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to the training images")
    ap.add_argument("-e", "--testing", required=True, help="path to the test images")
    args = vars(ap.parse_args())

    # get lbp features
    # (labels, data) = get_LBP_Features(args["training"])

    
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    losses_array = []
    accuracy_array = []
    for value in lr:
        print ("Running model for learning rate: {}".format(value))
        loss_array, accuracy = main(value, args["training"], args["testing"])
        losses_array.append(loss_array)
        accuracy_array.append(accuracy)
        print accuracy
        plt.plot(loss_array)
        plt.show()
    plt.plot(lr,accuracy_array)
    plt.show()
    print (accuracy_array)