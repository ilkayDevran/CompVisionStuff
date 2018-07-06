"""
https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/

--image  switch is the path to our input image that we want to detect pedestrians in.
--win-stride  is the step size in the x and y direction of our sliding window. 
--padding  switch controls the amount of pixels the ROI is padded 
    with prior to HOG feature vector extraction and SVM classification. 
To control the scale of the image pyramid (allowing us to detect people in images at multiple scales), 
    we can use the --scale  argument. 
--mean-shift  can be specified if we want to apply mean-shift grouping to the detected bounding boxes.


-- Default USAGE
    $ python main_hog.py --image images/person_010.bmp

--The smaller winStride  is, the more windows need to be evaluated 
(which can quickly turn into quite the computational burden):
    $ python main_hog.py --image images/person_010.bmp --win-stride="(4, 4)"

-- winStride  is the less windows need to be evaluated 
(allowing us to dramatically speed up our detector). 
However, if winStride  gets too large, then we can easily miss out on detections entirely:
    $ python main_hog.py --image images/person_010.bmp --win-stride="(16, 16)"

-- A smaller scale  will increase the number of layers in the image pyramid 
and increase the amount of time it takes to process your image:
    $ python detectmultiscale.py --image images/person_010.bmp --scale 1.01
"""

from __future__ import print_function
import argparse
import datetime
from imutils import paths
import imutils
import cv2
 

def get_hog_features(trainingSetPath, win_stride, padding, mean_shift, scale):
    # evaluate the command line arguments (using the eval function like
    # this is not good form, but let's tolerate it for the example)
    winStride = eval(win_stride)
    padding = eval(padding)
    meanShift = True if mean_shift > 0 else False
    
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # CHECK FOR IT!

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
        rects, weights = hog.detectMultiScale(img, winStride=winStride,
            padding=padding, scale=scale, useMeanshiftGrouping=meanShift)
        print (rects)
        # get hog features
        """STAYED HERE!"""

		# extract the label from the image path, then update the
		# label and data lists
        labels.append(imagePath.split("/")[-2])
        data.append(weights)
        


   
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True,
		help="path to training dataset")
    ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
        help="window stride")
    ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
        help="object padding")
    ap.add_argument("-s", "--scale", type=float, default=1.05,
        help="image pyramid scale")
    ap.add_argument("-m", "--mean-shift", type=int, default=-1,
        help="whether or not mean shift grouping should be used")
    args = vars(ap.parse_args())

    get_hog_features(args["training"], args["win_stride"], args["padding"],args["mean_shift"], args["scale"])