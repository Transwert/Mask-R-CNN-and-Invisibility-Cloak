# Import essential libraries
import cv2
import time
import numpy as np
from matplotlib import pyplot
import colorsys
import random

from mrcnn import visualize
from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
 
# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80
 
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(img, r, COLORS):
    """Apply the given mask to the image.
    """
    
    for i in range(0, r["rois"].shape[0]):
        #extract the class ID and mask for the current detection, then
        #grab the color to visualize the mask (in BGR format)
        classID = r["class_ids"][i]
        mask = r["masks"][:, :, i]
        color = COLORS[classID][::-1]
        #visualize the pixel-wise mask of the object
        img = visualize.apply_mask(img, mask, color, alpha=0.5)
    
    
    return img

white = (255, 255, 255)
hsv = [(i / len(class_names), 1, 1.0) for i in range(len(class_names))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

# Capturing from web cam
cap = cv2.VideoCapture(0)

# Preparation for writing the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Give the camera time(in seconds) for warming up
time.sleep(3)
count = 0
background = 0

# Capture the background in range of 60
# So as to give it some time to capture the background properly
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

# Read every frame from the web cam
while (cap.isOpened()):
    ret, image = cap.read()

    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    count += 1
    image = np.flip(image, axis=1)

    # Convert the image from BGR to HSV for better detection of red color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    
    results = rcnn.detect([image], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    image = apply_mask(image, r, COLORS)

    # Ranges should be carefully chosen
    # Setting the lower and upper range of mask1
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Setting the lower and upper range of mask2
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    # Mask refinement corresponding to the detected red color
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverted mask to segment out the red color from the frame
    mask2 = cv2.bitwise_not(mask1)

    # Segment the red color part out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(image, image, mask=mask2)

    # Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    # Generating the final output
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    output.write(finalOutput)
    cv2.imshow("magic", finalOutput)
    cv2.waitKey(1)

# Release the capture at the last
cap.release()
output.release()
cv2.destroyAllWindows()
