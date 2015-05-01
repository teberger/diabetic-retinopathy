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

# Standard resolution of images after processing.
STD_WIDTH = 512
STD_HEIGHT = 512

# Thresholding parameter. Determined experimentally.
THRESH = 22


def experiment_threshold(path):
    """
    Launches experiment window for thresholding.
    :param str path: Path to the experiment image file.
    """
    img_in = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.equalizeHist(img_in, img_in)
    ratio = float(STD_HEIGHT) / img_in.shape[0]  # Resize based on height.
    img_resize = cv.resize(img_in, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    _, thresh_img = cv.threshold(img_resize, 0, 255, cv.THRESH_BINARY)

    # Image windows for this experiment.
    t_window = "Threshold Experiment"
    o_window = "Original Image"
    cv.namedWindow(t_window, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(o_window, cv.WINDOW_AUTOSIZE)

    # Callback for parameter slider (instantiated afterward).
    def thresh_callback(pos):
        cv.threshold(img_resize, pos, 255, cv.THRESH_BINARY, dst=thresh_img)
        cv.imshow(t_window, thresh_img)
        return

    # Create the experiment and original image windows.
    cv.createTrackbar("Threshold", t_window, 0, 255, thresh_callback)
    cv.imshow(t_window, thresh_img)
    cv.imshow(o_window, img_resize)
    cv.waitKey(0)
    return


# def experiment_hough(path):
#     """
#     Launches experiment window for the Hough Circle Transformation.
#     :param str path: Path to the experiment image file.
#     """
#     img_in = cv.imread(path, cv.IMREAD_GRAYSCALE)
#     cv.equalizeHist(img_in, img_in)
#     ratio = float(STD_HEIGHT) / img_in.shape[0]  # Resize based on height.
#     img_resize = cv.resize(img_in, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
#     _, thresh_img = cv.threshold(img_resize, THRESH, 255, cv.THRESH_BINARY)
#
#     # Image windows for this experiment.
#     t_window = "Hough Circle Transform Experiment"
#     cv.namedWindow(t_window, cv.WINDOW_AUTOSIZE)
#
#     # Callback for parameter slider (instantiated afterward).
#     def thresh_callback(pos):
#         cv.threshold(img_resize, pos, 255, cv.THRESH_BINARY, dst=thresh_img)
#         cv.imshow(t_window, thresh_img)
#         return
#
#     cv.HoughCircles(img_resize, cv.HOUGH_STANDARD, 1, 300)
#
#     # Create the experiment and original image windows.
#     cv.createTrackbar("Threshold", t_window, 0, 255, thresh_callback)
#     cv.imshow(t_window, thresh_img)
#     cv.imshow(o_window, img_resize)
#     cv.waitKey(0)


if __name__ == "__main__":
    experiment_threshold("data/train/1000_left.jpeg")
