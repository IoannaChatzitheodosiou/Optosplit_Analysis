#Importing Relevant Libraries
import pandas as pd 
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt
import os
from read_roi import read_roi_zip
from pystackreg import StackReg
import tifffile
import sys

def calculate_mean_value(image, id, contour_threshold=50, show_image = False):
    ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh_8bit = cv2.convertScaleAbs(thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if len(contour)>contour_threshold]
    thecontours=cv2.drawContours(image, contours, -1, (0,255,0),  2)
    mean_values = []

    # Iterate through each contour (particle)
    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Extract pixels within the contour
        pixels = image[mask == 255]
        pixels = pixels[pixels != 0]

        # Calculate median value
        mean_value = np.mean(pixels)
        if mean_value==0:
            print(pixels)
        mean_values.append(mean_value)

    if show_image:
        plt.imshow(thecontours)
        plt.show()
    return mean_values

open_image = plt.imread("try/RML_split_beam_10_divided.tif")
means = calculate_mean_value(open_image, 1)