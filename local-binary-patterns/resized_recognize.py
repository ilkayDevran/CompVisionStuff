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
from sklearn.utils import shuffle
from sklearn import svm, grid_search



def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--training", required=True,
		help="path to the training images")
	ap.add_argument("-e", "--testing", required=True, 
		help="path to the tesitng images")
	args = vars(ap.parse_args())

	# initialize the local binary patterns descriptor along with the data and label lists
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
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)

		# extract the label from the image path, then update the
		# label and data lists
		labels.append(imagePath.split("/")[-2])
		data.append(hist)


	# shuffle datas and labels before trained a model
	X = np.array(data)
	y = np.array(labels)
	X, y = shuffle(X,y)

	# find best parameters before trained a model
	best_parameters = svc_param_selection(X,y)
	best_C = best_parameters.get("C")
	best_gamma =  best_parameters.get("gamma")
	print best_C, best_gamma
	# raw_input()

	# train a Linear SVM on the data
	model = LinearSVC(C=best_C, random_state=42)
	model.fit(X, y)


	# loop over the testing images
	for imagePath in paths.list_files(args["testing"], validExts=(".png",".ppm")):

		# load the image, convert it to grayscale, describe it, and classify it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)

		# prediction for each test image
		prediction = model.predict([hist])[0]

		# prepare data to append into lists
		inpt = str(imagePath.split("/")[-2])
		outpt = str(prediction)
		y_act.append(int(inpt)) 
		y_pred.append(int(outpt))


	# calculate match_count and accuracy to sho test result
	match_count = sum([int(y==y_) for y, y_ in zip(y_act, y_pred)])
	accuracy = float(match_count) / float(len(y_act))
	print "\nAccuracy:" + str(accuracy) + "\n"
	
	# get the experiment results
	# calculateConfusionMatrix(inp=y_act, out=y_pred)


def svc_param_selection(X, y, nfolds=None):
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 0.001,2000.0]
    gammas = [0.001, 0.01, 0.1, 1, 10, 20, 30]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_sr = grid_search.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_sr.fit(X, y)
    return grid_sr.best_params_


# show results of Confusion Matrix calculations
def calculateConfusionMatrix(inp, out):
	cm = ConfusionMatrix(actual_vector=inp, predict_vector=out)
	# LBP_ROI_Resized_Intensity_Normalization
	cm.save_html("LBP_ROI_Resized_Intensity_Normalization")
	print(cm)


if __name__ == '__main__':
    main()