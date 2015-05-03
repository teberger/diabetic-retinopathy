#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Image preprocessing function and experiments for parameter tuning. """

'''
=========
= NOTES =
=========

First normalize the image by removing color and auto-leveling.
--------------------------------------------------------------
    grayscale -> equalize histogram

Might want to downsize the images at this point to increase processing speed,
but reduced information available to rest of the operations.

Determine if image is inverted by detecting notch.
--------------------------------------------------
    threshold -> Hough circle -> paint largest circle -> blob detect

Idea is to find largest circle based on Hough algo which will be entire eye,
remove that circle from the thresholded image by painting it out, then doing
a blob detect on what's left. If there was a notch, there will be a blob left
with some mass that will exceed a threshold, and thus the image is inverted.

Scale and aspect ratio transformations.
---------------------------------------
    threshold -> bounding box -> extract bounded image ->
    scale largest axis of bounded image to standard resolution ->
    pad shorter axes with background

This step should make each eye image roughly the same size, regardless of the
scale the image started at. The big problem with this is that because many of
the images are cut off on the top and bottom, the resulting images will be
square images with large black bars on the top and bottom of the eyeball.

Image inversion step.
---------------------
    Image is not inverted:
        Image is left eye:
            Flip horizontal axis
    Image is inverted:
        Image is right eye:
            Flip both axes
        Image is left eye:
            Flip vertical axis

This step will set every eye image to face the same direction.

Feature extraction steps.
-------------------------
Methods to transform images into something the CNN can use to learn from besides
the regular grayscale versions. These methods are to be researched.

First method: PCA.
'''

import cv2 as cv
import numpy as np

# Standard resolution of images after processing.
STD_WIDTH = 512
STD_HEIGHT = 512

# Thresholding parameter.
# THRESH = 22
THRESH = 12

# Hough Circle Transform parameters.
DP = 1  # Inverse accumulator ratio.
MD = STD_HEIGHT  # Minimum distance between circles.
P1 = 140
P2 = 30
MIN_R = int(STD_HEIGHT * 0.4)
MAX_R = STD_HEIGHT


def load_image(path, grayscale=True, equalize=False, resize=True):
    """
    Loads an image, transforms it to grayscale, and resizes it. Optionally
    equalizes the image's histogram. Equalization seems to play poorly with
    preprocessing however, so by default it is turned off.
    :param path: Path to the image file.
    :param grayscale: Flag for converting image to grayscale.
    :param equalize: Flag for equalizing the image's histogram.
    :param resize: Flag for resizing the image to standard resolution.
    :rtype: np.ndarray
    """

    img_in = cv.imread(path, cv.IMREAD_GRAYSCALE if grayscale else -1)
    if equalize:
        cv.equalizeHist(img_in, img_in)

    if resize:
        ratio = float(STD_HEIGHT) / img_in.shape[0]  # Resize based on height.
        img_in = cv.resize(img_in, None,
                           fx=ratio,
                           fy=ratio,
                           interpolation=cv.INTER_AREA)

    return img_in


def threshold(img):
    """ Thresholds image according to global parameter. """
    _, output = cv.threshold(img, THRESH, 255, cv.THRESH_BINARY)
    return output


def hough_circles(img):
    """
    Apply Hough Circle Transform using global parameters and returns data in
    a nice list-of-tuples format. If no circles are found, the empty list is
    returned.
    :param np.ndarray img: The image to search for circles.
    :returns: List of tuples of the form (x, y, radius)
    :rtype: list[(int, int, float)]
    """
    global DP, MD, P1, P2, MIN_R, MAX_R

    #TODO TESTING
    h, w = img.shape
    half_h, half_w = (int(h / 2), int(w / 2))

    # cv.rectangle(img, (0, 0), (half_w, half_h),
    #              (0, 0, 0), thickness=cv.FILLED)
    # cv.rectangle(img, (half_w, half_h), (w, h),
    #              (0, 0, 0), thickness=cv.FILLED)

    cv.rectangle(img, (0, 0), (half_w, h),
                 (0, 0, 0), thickness=cv.FILLED)
    cv.rectangle(img, (0, h), (w, half_h),
                 (0, 0, 0), thickness=cv.FILLED)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, DP, MD,
                              param1=P1, param2=P2,
                              minRadius=MIN_R, maxRadius=MAX_R)
    #TODO TESTING

    output = []
    if circles is not None:
        circles = circles[0]  # Why are tuples buried like this? Bug?
        for c in circles[0:]:
            output.append((c[0], c[1], c[2]))

    return output


def draw_hough_circles(img, circles):
    """
    Creates new image from img with circles drawn over it. Will convert the
    output image to RGB space.
    :param np.ndarray img:
    :param list[(int, int, float)] circles: Detected circles.
    :rtype: np.ndarray
    """
    if circles:
        output = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)
        for c in circles:
            x, y, r = c
            cv.circle(output, (x, y), r, (0, 0, 255), 2)
    else:
        output = img

    return output


def experiment_threshold(path):
    """
    Launches experiment window for thresholding.
    :param str path: Path to the experiment image file.
    """
    img = load_image(path)
    _, thresh_img = cv.threshold(img, THRESH, 255, cv.THRESH_BINARY)

    # Image windows for this experiment.
    t_window = "Threshold Experiment"
    o_window = "Original Image"
    cv.namedWindow(t_window, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(o_window, cv.WINDOW_AUTOSIZE)

    # Callback for parameter slider (instantiated afterward).
    def thresh_callback(pos):
        cv.threshold(img, pos, 255, cv.THRESH_BINARY, dst=thresh_img)
        cv.imshow(t_window, thresh_img)
        return

    # Create the experiment and original image windows.
    cv.createTrackbar("Threshold", t_window, THRESH, 255, thresh_callback)
    cv.imshow(t_window, thresh_img)
    cv.imshow(o_window, img)
    cv.waitKey(0)
    return


def experiment_hough(path):
    """
    Launches experiment window for the Hough Circle Transformation.
    :param str path: Path to the experiment image file.
    """
    # Threshold the image first to get rid of pesky noise.
    img = load_image(path)

    # Image windows for this experiment.
    t_window = "Hough Circle Transform Experiment"
    cv.namedWindow(t_window, cv.WINDOW_AUTOSIZE)

    # Callbacks for parameter sliders (instantiated afterward).
    def dp_callback(pos):
        global DP
        DP = pos
        cv.imshow(t_window, draw_hough_circles(img, hough_circles(img)))
        return

    def p1_callback(pos):
        global P1
        P1 = pos
        cv.imshow(t_window, draw_hough_circles(img, hough_circles(img)))
        return

    def p2_callback(pos):
        global P2
        P2 = pos
        cv.imshow(t_window, draw_hough_circles(img, hough_circles(img)))
        # cv.imshow(t_window, hough_draw(img))
        return

    # Create the experiment and original image windows.
    cv.createTrackbar("DP", t_window, DP, 10, dp_callback)
    cv.createTrackbar("P1", t_window, P1, 255, p1_callback)
    cv.createTrackbar("P2", t_window, P2, 255, p2_callback)
    cv.imshow(t_window, draw_hough_circles(img, hough_circles(img)))
    cv.waitKey(0)


def experiment_notch_detection(path):
    # Get thresholded image and hough circles.
    img = load_image(path)
    img_thresh = threshold(img)
    circles = hough_circles(img)

    # Paint out the first circle detected. Assume that only one circle was
    # detected for whole image.
    x, y, r = circles[0]
    cv.circle(img_thresh, (x, y), r, (0, 0, 0), cv.FILLED)

    # Paint out the NW, SW, and SE quadrants of the image as only the
    # NE quadrant will contain a notch if present.
    h, w = img_thresh.shape
    half_h, half_w = (int(h / 2), int(w / 2))
    cv.rectangle(img_thresh, (0, 0), (half_w, h),
                 (0, 0, 0), thickness=cv.FILLED)
    cv.rectangle(img_thresh, (0, h), (w, half_h),
                 (0, 0, 0), thickness=cv.FILLED)

    # Erode what's left to try and remove edges.
    img_thresh = cv.erode(img_thresh, np.ones((3, 3), np.uint8))
    img_thresh = cv.dilate(img_thresh, np.ones((3, 3), np.uint8))

    # Run blob detection on what's left.
    # Set up the detector with default parameters.
    blob_params = cv.SimpleBlobDetector_Params()
    blob_params.minThreshold = 0.0
    blob_params.maxThreshold = THRESH
    blob_params.thresholdStep = THRESH / 2
    blob_params.filterByArea = False
    blob_params.filterByColor = False
    blob_params.filterByConvexity = False
    # blob_params.filterByInertia = False
    blob_params.minInertiaRatio = 0.05
    blob_params.maxInertiaRatio = 1
    sbd = cv.SimpleBlobDetector_create(blob_params)

    # Detect blobs.
    keypoints = sbd.detect(img_thresh)

    # Draw circles around any detected blobs.
    # cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle
    # corresponds to the size of blob
    img_thresh = cv.drawKeypoints(img_thresh, keypoints, np.array([]),
                                  (0, 0, 255),
                                  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Image windows for this experiment.
    o_window = "Original Image"
    t_window = "Thresholded, Subtracted, Blob-Detected"
    cv.namedWindow(o_window, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(t_window, cv.WINDOW_AUTOSIZE)
    cv.imshow(o_window, img)
    cv.imshow(t_window, img_thresh)
    cv.waitKey(0)
    return


if __name__ == "__main__":
    import os
    dir_path = "data/train"
    for file_path in os.listdir(dir_path):
        path = "{}/{}".format(dir_path, file_path)
        # experiment_threshold(path)
        experiment_hough(path)
        # experiment_notch_detection(path)
