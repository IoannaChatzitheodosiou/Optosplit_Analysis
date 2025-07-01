
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
import csv
from scipy.ndimage import gaussian_filter
from skimage import exposure

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

def normalize_image(image):
    image = image.astype(np.float32)
    image -= np.min(image)
    image /= (np.max(image) + 1e-5)
    return image

def ecc_align_images(reference, target, warp_mode=cv2.MOTION_AFFINE, number_of_iterations=5000, termination_eps=1e-6):
    # Convert images to float32 grayscale
    ref_float = reference.astype(np.float32)
    tgt_float = target.astype(np.float32)

    # Normalize both images
    ref_norm = cv2.normalize(ref_float, None, 0, 1, cv2.NORM_MINMAX)
    tgt_norm = cv2.normalize(tgt_float, None, 0, 1, cv2.NORM_MINMAX)

    # Define the motion model
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        cc, warp_matrix = cv2.findTransformECC(ref_norm, tgt_norm, warp_matrix, warp_mode, criteria)
    except cv2.error as e:
        print("ECC alignment failed:", e)
        return target  # fallback: return unaligned target

    # Warp the target image
    sz = reference.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(target, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(target, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return aligned

def image_stacking (cropped_high, cropped_low): #stacks the images
    aligned_low = ecc_align_images(cropped_high, cropped_low)

    # Combine the two crops into a 2-channel grayscale image
    try:
        combined_image = np.stack((cropped_high, aligned_low), axis=-1)  # Use np.stack to maintain grayscale
    except:
        print(cropped_high.shape)
        print(aligned_low.shape)

    split1 = combined_image[:, :, 0]
    split2 = combined_image[:, :, 1]

    combined = np.array([split1,split2], dtype=np.uint16)

    combined = combined.astype(np.uint16)

    aligned_low[aligned_low==0]= np.median (aligned_low)
    divided = aligned_low *1000/ cropped_high
    polarized_start = (aligned_low - cropped_high) / (aligned_low + cropped_high)
    #polarized_clipped = np.clip(polarized_start, -1, 1)
    gamma = 1.1
    polarized = (np.power(polarized_start, gamma)* 255).astype(np.uint8)
    divided = divided.astype(np.uint16)
    divided2 = cropped_low *1000/ cropped_high
    divided2 = divided2.astype(np.uint16)
    return combined, divided, polarized, divided2
 

def process_tif(image_name, image, outputpath, roi_high, roi_low):
    image_folder = image_name.replace(".tif","")
    if not os.path.exists(outputpath + "/" + image_folder):
        os.makedirs(outputpath + "/" + image_name.replace(".tif",""))
    image_low = apply_crop(roi_low, image)
    image_high = apply_crop(roi_high, image)
    lowname = outputpath + "/" +  image_folder + '/' + image_folder  + "_low.tif"
    highname = outputpath + "/" + image_folder + '/' + image_folder + "_high.tif"
    stackname = outputpath + "/" + image_folder + '/' + image_folder + "_stack.tif"
    divname = outputpath + "/" + image_folder + '/' + image_folder + "_divided.tif"
    divname2 = outputpath + "/" + image_folder + '/' + image_folder + "_divided_non_aligned.tif"
    polname = outputpath + "/" + image_folder + '/' + image_folder + "_polarized.tif"
    tifffile.imwrite(lowname, image_low, imagej=True)
    tifffile.imwrite(highname, image_high, imagej=True)
    stacked_image, divided, polarized, divided2 = image_stacking(image_high, image_low)
    tifffile.imwrite(stackname, stacked_image, imagej=True)
    tifffile.imwrite(divname, divided, imagej=True)
    tifffile.imwrite(divname2, divided2, imagej=True)
    tifffile.imwrite(polname, polarized, imagej=True)
    return polarized

def calculate_mean_value(image, image_name, output_path, contour_threshold=50):
    image_folder = image_name.replace(".tif","")
    if not os.path.exists(output_path + "/" + image_folder):
        os.makedirs(output_path + "/" + image_name.replace(".tif",""))
    contourname = output_path + "/" +  image_folder + '/' + image_folder  + "_contoured.tif"
    ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8) 
    eroded = cv2.erode(thresh, kernel, iterations=4) #morphological erosion to decrease particle size
    thresh_8bit = cv2.convertScaleAbs(eroded)

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

        # Calculate mean value
        mean_value = np.mean(pixels)
        if not np.isnan(mean_value):
            mean_values.append(mean_value)
        

    tifffile.imwrite(contourname, thecontours, imagej=True)
    return mean_values

def main():
    if len(sys.argv)!=4:
        print(f'Number of arguments expected: 4. Number of arguments provided {len(sys.argv)}')
        images_path = input('Images path: ')
        roipath = input('ROI path: ')
        output_path = input('Output path: ')
    else:
        images_path = sys.argv[1]
        roipath = sys.argv[2]
        output_path = sys.argv[3]
    roi_high, roi_low = roi_selection(roipath)
    images = image_listing(images_path)
    mean_values = []
    for image_name, image in images.items():
        mean_values.extend(calculate_mean_value(process_tif(image_name,image, output_path, roi_high, roi_low), image_name, output_path))
    with open(f'{output_path}/mean_values.csv', 'w') as f:
        for value in mean_values:
            f.write(str(value))
            f.write(';')
                           

if __name__ == "__main__":
    main()

