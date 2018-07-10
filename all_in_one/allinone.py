# USAGE
# python allinone.py -t ../ROI_images/training -e ../ROI_images/testing
# python allinone.py -t ../ROI_images/training -e ../ROI_images/testing -k 10 -j -1 -r 8 -p 24

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os



########### FEATURE EXTRACTION PART ##############

# GET LBP FEATURES 
def get_LBP_Features(trainingPath, testingPath, p, r):
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

	data = np.array(data)
	labels = np.array(labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

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
def get_HOG_Features(trainingPath, testingPath, cell_size, bin_size):
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
		""""
		print v.shape
		print len(vector)
		for i in v:
			print i
		# STAYED HERE VECTOR SHAPE IS 2D I need to make it 1D with a way
		raw_input()"""
		#print len(vector)
		# extract the label from the image path, then update the
		# label and data lists
		labels.append(imagePath.split("/")[-2])
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
		test_labels.append(imagePath.split("/")[-2])
		test_data.append(vector)

	data = np.array(data)
	labels = np.array(labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

	print data.shape

	print "[INFO] HOG Features are ready!"
	raw_input()
	return (data, labels, test_data, test_labels)


########### CLASSIFIER PART ##############

# USE kNN Classifier
def kNN(data, labels, test_data, test_labels, neighbors, jobs):

	# train and evaluate a k-NN classifer
	model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
	model.fit(data, labels)
	acc = model.score(test_data, test_labels)
	print("[INFO] accuracy: {:.2f}%".format(acc * 100))

# USE SVM Classifier
def SVM(data, labels, test_data, test_labels, best_kernel='linear', best_gamma='30'):
	from sklearn import svm

	# train a Linear SVM on the data
	model = svm.SVC(C=852.25, kernel=best_kernel)
	model.fit(data, labels)

	predicted=[]
	for d in test_data:
		p = model.predict([d])[0]
		predicted.append(p)
		
	match_count = sum([int(y==y_) for y, y_ in zip(test_labels, predicted)])
	acc = float(match_count) / float(len(test_labels))
	print("[INFO] accuracy: {:.2f}%".format(acc * 100))



### FEATURE DATA ORGANIZER TO PLOT PCA ###

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

# GET SIFT FEATURES FOR TRAINING DATA SET 
def get_all_samples_SIFT(training):
	all_samples, labels, samples_amount_of_classes = 0,0,0 #delete this row
	return all_samples, labels, samples_amount_of_classes

# GET HOG FEATURES FOR TRAINING DATA SET 
def get_all_samples_HOG(training):
	all_samples, labels, samples_amount_of_classes = 0,0,0 #delete this row
	return all_samples, labels, samples_amount_of_classes



########### PCA PART ##############

# SKLEARN TO GET PCA
def use_sklearn(all_samples, labels, samples_amount_of_classes,p=24,r=8, plot_it=False):
    from sklearn.decomposition import PCA as sklearnPCA
    from matplotlib import pyplot as plt

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



# CHOOSE THE RUNNING MOD OF THE SCRIPT
def chooseRunningMod(x, training, testing , neighbors, jobs, radius, points, cell_size, bin_size):
	if x == 1: 
		print "\n[INFO] LBP-KNN"
		(data, labels, test_data, test_labels) = get_LBP_Features(training, testing, p=points, r=radius)
		kNN(data, labels, test_data, test_labels, neighbors, jobs)

	elif x == 2:
		print "\n[INFO] SIFT-KNN"
		(key_points, descriptors, labels, test_key_points, test_descriptors, test_labels) = get_SIFT_Features(training, testing)
		kNN(descriptors, labels, test_descriptors, test_labels, neighbors, jobs)

	elif x == 3:
		print "\n[INFO] HOG-KNN"
		(data, labels, test_data, test_labels) = get_HOG_Features(training, testing, cell_size, bin_size)
		kNN(data, labels, test_data, test_labels, neighbors, jobs)
	elif x == 4:
		print "\n[INFO] LBP-SVM"
		(data, labels, test_data, test_labels) = get_LBP_Features(training, testing, p=points, r=radius)
		SVM(data, labels, test_data, test_labels)
	elif x == 5:
		print "\n[INFO] SIFT-SVM"

	elif x == 6:
		print "\n[INFO] HOG-SVM"
		(data, labels, test_data, test_labels) = get_HOG_Features(training, testing, cell_size, bin_size)
		SVM(data, labels, test_data, test_labels)

	elif x == 7:
		print "\n[INFO] PCA of LBP"
		all_samples, labels, samples_amount_of_classes = get_all_samples_LBP(training, p=points, r=radius)
		use_sklearn(all_samples, labels, samples_amount_of_classes, plot_it=True)

	elif x == 8:
		print "\n[INFO] PCA of SIFT"
		all_samples, labels, samples_amount_of_classes = get_all_samples_SIFT(training)
		use_sklearn(all_samples, labels, samples_amount_of_classes, plot_it=True)

	elif x == 9:
		print "\n[INFO] PCA of HOG"
		# get all_samples list
		all_samples, labels, samples_amount_of_classes = get_all_samples_HOG(training)
		use_sklearn(all_samples, labels, samples_amount_of_classes, plot_it=True)
	else:
		print "Please choose supported mods 1-7 to run this program."
		x = int(raw_input('>>> '))
		print "\n--[RESULTS]--"
		return chooseRunningMod(x, training, testing,neighbors, jobs, radius, points)



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
	ap.add_argument("-r", "--radius", type=int, default=8,
		help="radius parameter in LBP implementation")
	ap.add_argument("-p", "--points", type=int, default=24,
		help="radius parameter in LBP implementation")
	ap.add_argument("-c", "--cell_size", type=int, default=16,
		help="cell_size parameter in HOG implementation")
	ap.add_argument("-b", "--bin_size", type=int, default=8,
		help="bin_size parameter in HOG implementation")

	args = vars(ap.parse_args())

	print """
	Please choose the running mod you want between 1-9,
	Using k-NN as clasifier with:
		1. LBP
		2. SIFT
		3. HOG
	Using SVM as clasifier with:
		4. LBP
		5. SIFT
		6. HOG
	To see PCA of:
		7. LBP
		8. SIFT
		9. HOG
	"""
	x = int(raw_input('>>> '))
	print "\n--[RESULTS]--"
	chooseRunningMod(x,args["training"], args["testing"],args["neighbors"],
		args["jobs"], args["radius"], args["points"],args["cell_size"],args["bin_size"])