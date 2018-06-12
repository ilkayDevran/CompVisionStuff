# import the necessary packages
import numpy as np
import cv2
import argparse
import glob
import csv
import os
import argparse
from confusion import confusionMatrix as CM

class PaintingMatcher:
    def __init__(self, descriptor, paintingPaths, ratio = 0.7, minMatches = 40):
        self.descriptor = descriptor
        self.paintingPaths = paintingPaths   
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = "BruteForce"

    def search(self, queryKps, queryDescs):
        results = {}

        #counter = 0 # FOR DEBUGGING !!!!!!!!!!!!!!!!!!!!!**********

        # loop over the painting images
        for paintingPath in self.paintingPaths:

            #print counter, #FOR DEBUGGING

            # load the query image, convert it to grayscale, and
            # extract keypoints and descriptors
            painting = cv2.imread(paintingPath)
            gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)

            # determine the number of matched, inlier keypoints,
            # then update the results
            score = self.match(queryKps, queryDescs, kps, descs)
            results[paintingPath] = score

            #counter += 1 # FOR DEBUGGING
           

        # if matches were found, sort them
        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0], reverse=True)

        return results

    def match(self, kpsA, featuresA, kpsB, featuresB):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        if (len(kpsA) >= 2):
        	rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
	else:
		return -1.0
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        #print "Matches:" + str(len(matches))

        # check to see if there are enough matches to process
        if len(matches) > self.minMatches:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            # compute the homography between the two sets of points
            # and compute the ratio of matched points
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            # return the ratio of the number of matched keypoints
            # to the total number of keypoints
            return float(status.sum()) / status.size

        # no matches were found
        return -1.0

class PaintingDescriptor:

    def __init__(self):
        pass

    def describe(self, image): 
	descriptor = cv2.xfeatures2d.SIFT_create()

        # detect keypoints in the image, describing the region
        # surrounding each keypoint, then convert the keypoints
        # to a NumPy array
        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])

        return (kps, descs)


# FUNCTIONS OF MAIN IN BELOW

def getDefaultMatrix(h,w):
    return [[0 for c in range(w)] for r in range(h)] 

def setConfusionMatrix(input, output, matrix):
    print "input: " + input + "output: " + output
    matrix[int(input)][int(output)] += 1

# MAIN PART---- NEED TO BE EDDITTED
def main():

    # initialize the confusion matrix. Its size is depending on amount of Ground Truth
    GTAmount = os.listdir("groundTruth")
    #GTAmount.remove('.DS_Store') # remove returned hidden files by syscall
    confMatrix = CM(h = len(GTAmount), w = len(GTAmount))
    #confMatrix.toString()
    #confMatrix.getTheView()

    # initialize the database dictionary of groundTruth
    db = {}
    
    # loop over the database
    for l in csv.reader(open('database.csv')):
        # update the database using the image ID as the key
        db[l[0]] = l[1:]

    ratio = 0.7
    minMatches = 5

    #https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # initialize the paintingDescriptor and paintingMatcher
    pd = PaintingDescriptor()
    pm = PaintingMatcher(pd, glob.glob(os.path.join('groundTruth', '*.ppm')) , ratio = ratio, minMatches = minMatches)

    getDefaultMatrix(5,3)
    src = 'queries'

    src_files = os.listdir(src)
    #print len(src_files)
    count = 0
    for image in src_files:
	count = count+1
	print (count,len(src_files))
        print image
        full_file_name = os.path.join(src, image)
        # load the query image, convert it to grayscale, and extract
        # keypoints and descriptors
        queryImage = cv2.imread(full_file_name)  # the query image
        gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
        (queryKps, queryDescs) = pd.describe(gray)
        print "Number of keypoints for query:",str(len(queryKps))
        # try to match the book painting to a known database of images
        results = pm.search(queryKps, queryDescs)

        # check to see if no results were found
        if len(results) == 0:
            print("Cannot not find a match for that sign!")

        # otherwise, matches were found
        else:
            score, paintingPath = results[0]
            (classOfSign, classNo) = db[paintingPath[paintingPath.rfind("/") + 1:]]
            print("{}. {:.2f}% : {} - {}".format(1, score * 100,
                classOfSign, classNo))

            # Some string manipulation stuffs
            classNoInString = image.index('.')
            confMatrix.setConfusionMatrix(int(image[5:classNoInString]), int(classNo[1:]))
        print "\n"

    confMatrix.toString()
    confMatrix.getTheView()

if __name__ == '__main__':
    main()
