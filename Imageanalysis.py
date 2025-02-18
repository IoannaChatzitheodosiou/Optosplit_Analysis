
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

def image_listing(datapath):  #function that reads your images from the folder
    imagedata = datapath + "/*.tif" #how your image files look
    imagelist = glob(imagedata) #makes a list of all the images in your folder
    open_images = {} #dictionary for open images
    # Iterating over the list of images and opening them
    for image in imagelist:
        # Get the filename (without the full path)
        filename = os.path.basename(image)
        # Read the image and store it in the dictionary
        open_images[filename] = plt.imread(image)
    return (open_images)

def roi_selection(roipath): #function that reads the rois
    rois = read_roi_zip(roipath)
    roi_high = rois['high'] #higher wavelength roi
    roi_low = rois['low']  #low wavelength roi
    return (roi_high, roi_low)

def apply_crop(roi, image): #function that crops the image into two, according to the rois
    top_left_y = roi['top']
    top_left_x = roi['left']
    bottom_right_y = top_left_y + roi['height']
    bottom_right_x = top_left_x + roi['width']
    return image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

def image_stacking (cropped_high, cropped_low): #function that crops the images
    sr = StackReg(StackReg.BILINEAR)
    aligned_low = sr.register_transform(cropped_high, cropped_low)
    # Combine the two crops into a 2-channel grayscale image
    try:
        combined_image = np.stack((cropped_high, aligned_low), axis=-1)  # Use np.stack to maintain grayscale
        print(combined_image.shape)
    except:
        print(cropped_high.shape)
        print(aligned_low.shape)

    split1 = combined_image[:, :, 0]
    split2 = combined_image[:, :, 1]

    combined = np.array([split1,split2], dtype=np.uint16)

    combined = combined.astype(np.uint16)

    aligned_low[aligned_low==0]= np.median (aligned_low)
    divided = aligned_low *1000/ cropped_high
    divided = divided.astype(np.uint16)
    print (divided)
    print ()
    divided2 = cropped_low *1000/ cropped_high
    divided2 = divided2.astype(np.uint16)
    print (divided2)
    print ()
    return combined, divided, divided2
 

def process_tif(image_name, image, outputpath, roi_high, roi_low):
    image_low = apply_crop(roi_low, image)
    image_high = apply_crop(roi_high, image)
    lowname = outputpath + "/" + image_name.replace(".tif","") + "_low.tif"
    highname = outputpath + "/" + image_name.replace(".tif","") + "_high.tif"
    stackname = outputpath + "/" + image_name.replace(".tif","") + "_stack.tif"
    divname = outputpath + "/" + image_name.replace(".tif","") + "_divided.tif"
    divname2 = outputpath + "/" + image_name.replace(".tif","") + "_divided_non_aligned.tif"
    tifffile.imwrite(lowname, image_low, imagej=True)
    tifffile.imwrite(highname, image_high, imagej=True)
    stacked_image, divided, divided2 = image_stacking(image_high, image_low)
    tifffile.imwrite(stackname, stacked_image, imagej=True)
    tifffile.imwrite(divname, divided, imagej=True)
    tifffile.imwrite(divname2, divided2, imagej=True)


def main():
    if len(sys.argv)!=3:
        print(f'Number of arguments expected: 3. Number of arguments provided {len(sys.argv)}')
        images_path = input('Images path: ')
        roipath = input('ROI path: ')
        output_path = input('Output path: ')
    else:
        images_path = sys.argv[1]
        roipath = sys.argv[2]
        output_path = sys.argv[3]
    roi_high, roi_low = roi_selection(roipath)
    images = image_listing(images_path)
    for image_name, image in images.items():
        process_tif(image_name,image, output_path, roi_high, roi_low)

if __name__ == "__main__":
    main()

