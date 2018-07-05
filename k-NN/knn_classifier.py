# USAGE
# python knn_classifier.py -t /Users/ilkay/Desktop/git/CompVisionStuff/ROI_images/training -e /Users/ilkay/Desktop/git/CompVisionStuff/ROI_images/testing
# python knn_classifier.py --dataset kaggle_dogs_vs_cats -k 10

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


# GET LBP FEATURES 
def get_LBP_Features(trainingPath, testingPath, p=24, r=8):
	from localbinarypatterns import LocalBinaryPatterns

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
		labels.append(imagePath.split("/")[-2])
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
		test_labels.append(imagePath.split("/")[-2])
		test_data.append(hist)

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
def get_HOG_Features():
	pass


# USE kNN Classifier
def kNN(data, labels, test_data, test_labels, neighbors, jobs):
	# turn all list into numpy array
	data = np.array(data)
	labels = np.array(labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

	# train and evaluate a k-NN classifer
	model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
	model.fit(data, labels)
	acc = model.score(test_data, test_labels)
	print("[INFO] accuracy: {:.2f}%".format(acc * 100))


# USE SVM Classifier
def SVM(data, labels, test_data, test_labels):
	pass



########### PCA PART ##############

# SKLEARN TO GET PCA
def use_sklearn(all_samples, labels, samples_amount_of_classes,p=24,r=8, plot_it=False):
    from sklearn.decomposition import PCA as sklearnPCA

    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
    sklearn_transf = sklearn_transf.T

    max_Y = 0.
    max_X = 0.
    min_Y = 0.
    min_X = 0.

    temp = 0
    for i,val in enumerate(samples_amount_of_classes):

        max_X, min_X, max_Y, min_Y = find_max_min_X_Y(sklearn_transf[0,temp:val+temp],sklearn_transf[1,temp:val+temp],
            max_X, min_X, max_Y, min_Y)
        plt.plot(sklearn_transf[0,temp:val+temp],sklearn_transf[1,temp:val+temp], 'o', markersize=7, color=np.random.rand(3,), alpha=0.5, label=labels[i])
        temp = val
        # plt.show()
        # raw_input("Class name: {}".format(labels[i]))

    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.xlim([min_X,max_X])
    plt.ylim([min_Y, max_Y])
    #plt.legend()
    plt.title('-SKleanr- Points: '+ str(p) + " Radius:" + str(r))
    if plot_it == False:
        fname = '/Users/ilkay/Desktop/figures/sklearn/s_' + str(p) +"_" + str(r) + ".svg"
        plt.savefig(fname, format='svg')
    else:
        plt.show()
    print '-SKleanr- Points: '+ str(p) + ' Radius:' + str(r) + ' DONE!'


# Find max min X and Y for plotting
def find_max_min_X_Y(x, y, max_X, min_X, max_Y, min_Y):
    
    max_of_x_list = max(x)
    min_of_x_list = min(x)
    max_of_y_list = max(y)
    min_of_y_list = min(y)

    if max_of_x_list > max_X:
        max_X = max_of_x_list
    elif min_of_x_list < min_X :
        min_X = min_of_x_list

    if max_of_y_list > max_Y:
        max_Y = max_of_y_list
    elif min_of_y_list < min_Y :
        min_Y = min_of_y_list

    return (max_X, min_X, max_Y, min_Y)


# GET LBP FEATURES FOR TRAINING DATA SET 
def get_all_samples_LBP(path, p=24, r=8 ):
	from localbinarypatterns import LocalBinaryPatterns

	# initialize the local binary patterns descriptor along with the data and label lists
	desc = LocalBinaryPatterns(p, r)
	data = []
	labels = []
	classSamplesList = []
	samples_amount_of_classes = []
	currentClass = None
	flag = False

	class_list = os.listdir(path)
	class_list.remove('.DS_Store')
	class_list.remove('Readme.txt')
	counter = len(class_list)

	lastClassPath = ''
	# loop over the training images
	for imagePath in paths.list_files(path, validExts=(".png",".ppm")):
		if (flag == False):
			currentClass = imagePath.split("/")[-2]
			labels.append(currentClass)
			counter -= 1
			flag = True
		else:
			if imagePath.split("/")[-2] != currentClass:
				currentClass = imagePath.split("/")[-2]
				classSamplesList.append(np.transpose(np.array(data)))
				samples_amount_of_classes.append(len(data))
				data = []
				labels.append(currentClass)
				counter -= 1
		if counter == 0:
			lastClassPath = imagePath
			break
					
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)
		
		# extract the label from the image path
		data.append(hist)

	data = []
	head, _ = os.path.split(lastClassPath)

	for imagePath in paths.list_files(head, validExts=(".png", ".ppm")):
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		resized_image = cv2.resize(gray, (32, 32))
		hist = desc.describe(resized_image)
		hist = hist / max(hist)
		# extract the label from the image path
		data.append(hist)

	classSamplesList.append(np.transpose(np.array(data)))
	samples_amount_of_classes.append(len(data))


	all_samples =  tuple(classSamplesList)
	all_samples = np.concatenate(all_samples, axis=1)

	return all_samples, labels ,samples_amount_of_classes

# MAIN
if __name__ == '__main__':
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
	ap.add_argument("-c", "--classifier", type=str, default='knn',
		help="choose classifier knn, svm")
	args = vars(ap.parse_args())


	print "\n[INFO] LBP-KNN"
	(data, labels, test_data, test_labels) = get_LBP_Features(args["training"], args["testing"])
	kNN(data, labels, test_data, test_labels, args["neighbors"], args["jobs"])

	print "\n[INFO] SIFT-KNN"
	(key_points, descriptors, labels, test_key_points, test_descriptors, test_labels) = get_SIFT_Features(args["training"], args["testing"])
	for i, val in enumerate(descriptors):
		print len(val)
		print labels[i]
		raw_input()
	kNN(descriptors, labels, test_descriptors, test_labels, args["neighbors"], args["jobs"])

