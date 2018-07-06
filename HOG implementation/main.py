# USAGE
# python main.py -t ../images/training -e ../images/testing
# python main.py -t ../ROI_images/training -e ../ROI_images/testing

from hog import Hog_descriptor as descriptor
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import cv2

def main(trainingSetPath, testingSetPath, cell_size=8, bin_size=8):

    # initialize the local binary patterns descriptor along with the data and label lists
    labels = []
    data = []
    test_labels = []
    test_data = []

    # loop over the training images
    for imagePath in paths.list_files(trainingSetPath, validExts=(".png",".ppm")):
        
        # open image
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # resized_image = cv2.resize(img, (32, 32))  # RESIZING

        # get hog features
        hog = descriptor(img, cell_size=cell_size, bin_size=bin_size)
        vector, image = hog.extract()
        # vector = vector / max(vector)  # NORMALAZING 
        print len(vector)

		# extract the label from the image path, then update the
		# label and data lists
        labels.append(imagePath.split("/")[-2])
        data.append(vector)


    # loop over the testing images
    for imagePath in paths.list_files(testingSetPath, validExts=(".png",".ppm")):
        
        # open image
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # resized_image = cv2.resize(img, (32, 32))  # RESIZING

        # get hog features
        hog = descriptor(img, cell_size=cell_size, bin_size=bin_size)
        vector, image = hog.extract()
        # vector = vector / max(vector)  # NORMALAZING 
        print len(vector)

		# extract the label from the image path, then update the
		# label and data lists
        test_labels.append(imagePath.split("/")[-2])
        test_data.append(vector)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to the training images")
    ap.add_argument("-e", "--testing", required=True, help="path to the tesitng images")
    args = vars(ap.parse_args())

    main(args["training"], args["testing"], 8, 8)