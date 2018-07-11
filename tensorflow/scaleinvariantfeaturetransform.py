import numpy as np
import cv2

class ScaleInvariantFeatureTransform:
    def __init__(self):
        pass
    def describe(self, image): 
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, descs)