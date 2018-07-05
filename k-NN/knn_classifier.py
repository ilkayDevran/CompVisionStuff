# USAGE
# python knn_classifier.py -t /Users/ilkay/Desktop/git/CompVisionStuff/ROI_images/training -e /Users/ilkay/Desktop/git/CompVisionStuff/ROI_images/testing
# python knn_classifier.py --dataset kaggle_dogs_vs_cats -k 10

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from localbinarypatterns import LocalBinaryPatterns
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()

def chooseDescriptor(desc):
	if desc == 'sift':
		return
	elif desc == 'lbp':
		return LocalBinaryPatterns(24, 8)
	elif desc == 'hog':
		return

def kNN():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--training", required=True,
		help="path to training dataset")
	ap.add_argument("-e", "--testing", required=True,
		help="path to testing dataset")
	ap.add_argument("-k", "--neighbors", type=int, default=1,
		help="# of nearest neighbors for classification")
	ap.add_argument("-j", "--jobs", type=int, default=-1,
		help="# of jobs for k-NN distance (-1 uses all available cores)")
	ap.add_argument("-d", "--desc", type=str, default='lbp',
		help="choose feature extractor sift, lbp, hog")
	args = vars(ap.parse_args())

	data = []
	labels = []
	test_data= []
	test_labels = []

	# loop over the training images
	for imagePath in paths.list_files(args["training"], validExts=(".png",".ppm")):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg
		image = cv2.imread(imagePath)
		# extract raw pixel intensity "data", followed by a color
		# histogram to characterize the color distribution of the pixels
		# in the image
		"""EDIT HERE ACCORDING TO USED FEATURE EXTRACTOR"""
		hist = extract_color_histogram(image)

		# update the raw images, data, and labels matricies,
		# respectively
		data.append(hist)
		labels.append(imagePath.split("/")[-2])

	# loop over the testing images
	for imagePath in paths.list_files(args["testing"], validExts=(".png",".ppm")):
		# load the image and extract the class label (assuming that our
		# path as the format: /path/to/dataset/{class}.{image_num}.jpg
		image = cv2.imread(imagePath)
		# extract raw pixel intensity "data", followed by a color
		# histogram to characterize the color distribution of the pixels
		# in the image
		"""EDIT HERE ACCORDING TO USED FEATURE EXTRACTOR"""
		hist = extract_color_histogram(image)

		# update the raw images, data, and labels matricies,
		# respectively
		test_data.append(hist)
		test_labels.append(imagePath.split("/")[-2])

	# turn all list into numpy array
	data = np.array(data)
	labels = np.array(labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

	# train and evaluate a k-NN classifer on the raw pixel intensities
	print("[INFO] evaluating raw pixel accuracy...")
	model = KNeighborsClassifier(n_neighbors=args["neighbors"],
		n_jobs=args["jobs"])
	model.fit(data, labels)
	acc = model.score(test_data, test_labels)
	print("[INFO] accuracy: {:.2f}%".format(acc * 100))


if __name__ == '__main__':
    kNN()