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
from localbinarypatterns import LocalBinaryPatterns

from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder



########### FEATURE EXTRACTION PART ##############

# GET LBP FEATURES 
def get_LBP_Features(trainingPath, testingPath ,p=24, r=8):
	from localbinarypatterns import LocalBinaryPatterns
	from sklearn.utils import shuffle

	# initialize the local binary patterns descriptor along with the data and label lists
	desc = LocalBinaryPatterns(p, r)
	data = []
	labels = []
	test_data = []
	test_labels = []

	# loop over the training images
	for imagePath in paths.list_files(trainingPath, validExts=(".png",".ppm")):
		
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)

		# extract the label from the image path, then update the
		# label and data lists
		labels.append(int(imagePath.split("/")[-2]))
		data.append(hist)

	# loop over the testing images
	for imagePath in paths.list_files(testingPath, validExts=(".png",".ppm")):

		# load the image, convert it to grayscale, describe it, and classify it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)

		# extract the label from the image path, then update the
		# label and data lists
		test_labels.append(int(imagePath.split("/")[-2]))
		test_data.append(hist)

	data = np.array(data)
	labels = np.array(labels)
	#test_data = np.array(test_data)
	#test_labels = np.array(test_labels)

	data, labels = shuffle(data,labels)

	print "[INFO] LBP Features are ready!"

	return (data, labels, test_data, test_labels)

# GET SIFT FEATURES
def get_SIFT_Features(trainingPath, testingPath):
	from scaleinvariantfeaturetransform import ScaleInvariantFeatureTransform

	desc = ScaleInvariantFeatureTransform()

	key_points = []
	descriptors = []
	labels = []

	test_key_points = []
	test_descriptors = []
	test_labels = []

	# loop over the training images
	for imagePath in paths.list_files(trainingPath, validExts=(".png",".ppm")):
		
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		(kps, descs) = desc.describe(image)
		#print "Number of keypoints for image: ", str(len(kps))

		key_points.append(kps)
		descriptors.append(descs)
		labels.append(imagePath.split("/")[-2])

	# loop over the testing images
	for imagePath in paths.list_files(testingPath, validExts=(".png",".ppm")):
		
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		(kps, descs) = desc.describe(image)
		#print "Number of keypoints for image: ", str(len(kps))

		test_key_points.append(kps)
		test_descriptors.append(descs)
		test_labels.append(imagePath.split("/")[-2])

	print "[INFO] SIFT Features are ready!"

	return (key_points, descriptors, labels, test_key_points, test_descriptors, test_labels)

# GET HOG FEATURES
def get_HOG_Features(trainingPath, testingPath, cell_size=16, bin_size=8):
	from hog import Hog_descriptor

	# initialize the local binary patterns descriptor along with the data and label lists
	data = []
	labels = []
	test_data = []
	test_labels = []
    
	# loop over the training images
	for imagePath in paths.list_files(trainingPath, validExts=(".png",".ppm")):
		# open image
		img = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (48, 48))

		# get hog features
		hog = Hog_descriptor(resized_image, cell_size=cell_size, bin_size=bin_size)
		vector = hog.extract()
		v = np.array(vector)

		# extract the label from the image path, then update the
		# label and data lists
		labels.append(int(imagePath.split("/")[-2]))
		data.append(vector)

	# loop over the testing images
	for imagePath in paths.list_files(testingPath, validExts=(".png",".ppm")):
		
		# open image
		img = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (48, 48))

		# get hog features
		hog = Hog_descriptor(resized_image, cell_size=cell_size, bin_size=bin_size)
		vector = hog.extract()

		# extract the label from the image path, then update the
		# label and data lists
		test_labels.append(int(imagePath.split("/")[-2]))
		test_data.append(vector)

	data = np.array(data)
	labels = np.array(labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

	print "[INFO] HOG Features are ready!"

	return (data, labels, test_data, test_labels)



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

def main(lr, train_data_dir, test_data_dir, images_array, labels_array, test_images_, test_labels, placeholder_shape, num_of_epochs, log_freq):

    # define globals.
    SIZE_W = 32
    SIZE_H = 32
    CHANNELS = 3

    d_in = SIZE_W * SIZE_H * CHANNELS
    d_out = 62

    learning_rate = lr

    graph = tf.Graph()

    with graph.as_default():
        images_ph = tf.placeholder(tf.float32, placeholder_shape) # [None, SIZE_W, SIZE_H, CHANNELS] || [None, vector_lentgh]
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
    predicted = session.run([predicted_labels], feed_dict={images_ph: test_images_})[0]
    match_count = sum([int(y==y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = float(match_count) / float(len(test_labels))

    return loss_array, accuracy

    
    # create siggNet object.


# CHOOSE THE RUNNING MOD OF THE SCRIPT
def chooseRunningMod(x, train_data_dir, test_data_dir, radius, points, cell_size, bin_size, value, num_of_epochs, log_freq):
    if x == 1: 
        print "\n[INFO] Feature extractor -> LBP"
        images_array, labels_array, test_images_, test_labels = get_LBP_Features(train_data_dir, test_data_dir, p=24, r=8)
        
        vector_lentgh=len(images_array[0])
        placeholder_s = [None, vector_lentgh]
        
        print ("Running model for learning rate: {}".format(value))
        loss_array, accuracy = main(value, train_data_dir, test_data_dir, 
            images_array, labels_array, test_images_, test_labels, placeholder_shape=placeholder_s, num_of_epochs=num_of_epochs, log_freq=log_freq)
        print "\n[INFO] Accuracy:" + str(accuracy)
        plt.plot(loss_array)
        plt.show()

    elif x == 2:
        print "\n[INFO] Feature extractor -> HOG"
        images_array, labels_array, test_images_, test_labels = get_HOG_Features(train_data_dir, test_data_dir, cell_size=16, bin_size=8)
        
        vector_lentgh=len(images_array[0])
        placeholder_s = [None, vector_lentgh]

        print ("Running model for learning rate: {}".format(value))
        loss_array, accuracy = main(value, train_data_dir, test_data_dir, 
            images_array, labels_array, test_images_, test_labels, placeholder_shape=placeholder_s, num_of_epochs=num_of_epochs, log_freq=log_freq)
        print "\n[INFO] Accuracy:" + str(accuracy)
        plt.plot(loss_array)
        plt.show()
        
    elif x == 3:
        print "\n[INFO] Feature extractor -> SIFT"
        images_array, labels_array, test_images_, test_labels = get_SIFT_Features(train_data_dir, test_data_dir)
        
        vector_lentgh=len(images_array[0])
        placeholder_s = [None, vector_lentgh]

        print ("Running model for learning rate: {}".format(value))
        loss_array, accuracy = main(value, train_data_dir, test_data_dir, 
            images_array, labels_array, test_images_, test_labels, placeholder_shape=placeholder_s, num_of_epochs=num_of_epochs, log_freq=log_freq)
        print "\n[INFO] Accuracy:" + str(accuracy)
        plt.plot(loss_array)
        plt.show()

    else:
        print "Please choose supported mods 1-9 to run this program."
        x = int(raw_input('>>> '))
        print "\n--[RESULTS]--"
        return chooseRunningMod(x, train_data_dir, test_data_dir, radius, points, cell_size, bin_size)



if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to the training images")
    ap.add_argument("-e", "--testing", required=True, help="path to the test images")
    ap.add_argument("-r", "--radius", type=int, default=8, help="radius parameter in LBP implementation")
    ap.add_argument("-p", "--points", type=int, default=24, help="radius parameter in LBP implementation")
    ap.add_argument("-c", "--cell_size", type=int, default=16, help="cell_size parameter in HOG implementation")
    ap.add_argument("-b", "--bin_size", type=int, default=8, help="bin_size parameter in HOG implementation")
    ap.add_argument("-lr", "--learningR", type=int, default=1e-1, help="learning rate")
    ap.add_argument("-ep", "--epochs", type=int, default=10000, help="number of epochs")
    ap.add_argument("-l", "--freq", type=int, default=100, help="log freq")

    args = vars(ap.parse_args())

    print """
	Please choose the running mod you want between 1 - 4,
		1. LBP
		2. HOG
		3. SIFT
	"""
    x = int(raw_input('>>> '))
    print "\n--[RESULTS]--"
    chooseRunningMod(x, args["training"], args["testing"], 
        args["radius"], args["points"],args["cell_size"],args["bin_size"],args["learningR"], args["epochs"],args["freq"])

    """
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    losses_array = []
    accuracy_array = []
    for value in lr:
        print ("Running model for learning rate: {}".format(value))
        loss_array, accuracy = main(value, args["training"], args["testing"])
        losses_array.append(loss_array)
        accuracy_array.append(accuracy)
        print "\n[INFO] Accuracy:" + str(accuracy)
        plt.plot(loss_array)
        plt.show()
    plt.plot(lr,accuracy_array)
    plt.show()
    print (accuracy_array)
    """