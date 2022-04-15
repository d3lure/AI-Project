# import the necessary packages
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from detection_helpers import sliding_window
from detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to the input image")
# ap.add_argument("-s", "--size", type=str, default="(200, 150)",
#                 help="ROI size (in pixels)")
# ap.add_argument("-c", "--min-conf", type=float, default=0.9,
#                 help="minimum probability to filter weak detections")
# ap.add_argument("-v", "--visualize", type=int, default=-1,
#                 help="whether or not to show extra visualizations for debugging")
# args = vars(ap.parse_args())

# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (200, 150)
INPUT_SIZE = (224, 224)

# load our network weights from disk
print("[INFO] loading network...")
model = tf.keras.models.load_model('cnn_model.h5')

# load the input image from disk, resize it such that it has the supplied width, and then grab its dimensions
# orig = cv2.imread(args["image"])
orig = 'sunflower'
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]
