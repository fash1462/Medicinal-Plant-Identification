import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to the input image', 
                    required=True)
args = vars(parser.parse_args())

image = cv2.imread(args['input'])
# keep a copy of the original image
orig_image = image.copy()
# convert to RGB image and convert to float32
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
# grayscale and blurring for canny edge detection
gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# carry out Canny edge detection
# canny = cv2.Canny(blurred, 50, 200)
# initialize the structured edge detector with the model
edge_detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml/model.yml')
# detect the edges
edges = edge_detector.detectEdges(image)

# show and save the image edges
save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}"
cv2.imshow('Structured forests', edges)
# cv2.imshow('Canny', canny)
# cv2.waitKey(0)
cv2.imwrite(f"{save_name}_forests.jpg", edges*255.0)
# cv2.imwrite(f"{save_name}_canny.jpg", canny)