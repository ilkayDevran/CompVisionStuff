# USAGE
# python resized_recognize.py --training images/training --testing images/testing
# python resized_recognize.py --training ROI_images/training --testing ROI_images/testing

# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import numpy as np
from pycm import *


def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--training", required=True,
		help="path to the training images")
	ap.add_argument("-e", "--testing", required=True, 
		help="path to the tesitng images")
	args = vars(ap.parse_args())

	# initialize the local binary patterns descriptor along with
	# the data and label lists
	desc = LocalBinaryPatterns(24, 8)
	data = []
	labels = []

	# initialize the actual and predicted vectors
	y_act = []
	y_pred = []

	# loop over the training images
	for imagePath in paths.list_files(args["training"], validExts=(".png",".ppm")):
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		normalized = np.multiply(gray, 1.0/gray.max())
		resized_image = cv2.resize(normalized, (32, 32))
		hist = desc.describe(resized_image)

		# extract the label from the image path, then update the
		# label and data lists
		labels.append(imagePath.split("/")[-2])
		data.append(hist)

	# train a Linear SVM on the data
	model = LinearSVC(C=100.0, random_state=42)
	model.fit(data, labels)

	# loop over the testing images
	for imagePath in paths.list_files(args["training"], validExts=(".png",".ppm")):
		# load the image, convert it to grayscale, describe it,
		# and classify it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		normalized = np.multiply(gray, 1.0/gray.max())
		resized_image = cv2.resize(normalized, (32, 32))
		hist = desc.describe(resized_image)
		prediction = model.predict([hist])[0]

		inpt = str(imagePath.split("/")[-2])
		outpt = str(prediction)
		x = int(inpt)
		y = int(outpt)
		y_act.append(x) 
		y_pred.append(y)

	# get the experiment results
	calculateConfusionMatrix(inp=y_act, out=y_pred)

# show results of Confusion Matrix calculations
def calculateConfusionMatrix(inp, out):
	cm = ConfusionMatrix(actual_vector=inp, predict_vector=out)
	# LBP_ROI_Resized_Intensity_Normalization
	cm.save_html("LBP_ROI_Resized_Intensity_Normalization")
	print(cm)

if __name__ == '__main__':
    main()

