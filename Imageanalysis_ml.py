
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
import yaml


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

def crop_divided(divided):
    #define crop area
    height, width = divided.shape[:2]
    center_x = width // 2
    center_y = height // 2
    # Compute half-size crop bounds
    half_w = width // 4
    half_h = height // 4
    x1 = center_x - half_w
    x2 = center_x + half_w
    y1 = center_y - half_h
    y2 = center_y + half_h
    # Crop the image
    cropped = divided[y1:y2, x1:x2]
    return cropped

def image_stacking (cropped_high, cropped_low): 
    sr = StackReg(StackReg.AFFINE)
    aligned_high = sr.register_transform(cropped_low, cropped_high)
    # Combine the two crops into a 2-channel grayscale image
    try:
        combined_image = np.stack((aligned_high, cropped_low), axis=-1)  # Use np.stack to maintain grayscale
    except:
        print(cropped_low.shape)
        print(aligned_high.shape)

    split1 = combined_image[:, :, 0]
    split2 = combined_image[:, :, 1]

    combined = np.array([split1,split2], dtype=np.uint16)

    combined = combined.astype(np.uint16)

    aligned_high[aligned_high==0]= np.median (aligned_high)
    divided =   aligned_high*1500 / cropped_low
    divided = divided.astype(np.uint16)
    divided = crop_divided(divided)
    divided2 =  cropped_high*1500 /cropped_low 
    divided2 = divided2.astype(np.uint16)
    return combined, divided, divided2, aligned_high
    

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
    tifffile.imwrite(lowname, image_low, imagej=True)
    tifffile.imwrite(highname, image_high, imagej=True)
    stacked_image, divided, divided2, aligned_high = image_stacking(image_high, image_low)
    tifffile.imwrite(stackname, stacked_image, imagej=True)
    tifffile.imwrite(divname, divided, imagej=True)
    tifffile.imwrite(divname2, divided2, imagej=True)
    return aligned_high, image_low, divided

def calculate_mean_value_and_areas(divided_image, high_image, low_image, image_name, output_path, contour_threshold):
    areas_high = []
    areas_low = []
    image_folder = image_name.replace(".tif","")
    if not os.path.exists(output_path + "/" + image_folder):
        os.makedirs(output_path + "/" + image_name.replace(".tif",""))
    
    contourname = output_path + "/" +  image_folder + '/' + image_folder  + "_contoured.tif"
    
    norm_img = cv2.normalize(divided_image, None, 0, 255, cv2.NORM_MINMAX)
    divided_normalized = norm_img.astype('uint8')
    #Adaptive thresholding
    thresh = cv2.adaptiveThreshold(divided_normalized,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=51,  # must be odd and tailored to feature size
                                    C=5)
    
    kernel = np.ones((3, 3), np.uint8) 
    eroded = cv2.erode(thresh, kernel, iterations=1) #morphological erosion to decrease particle size
    kernel = np.ones((4, 4), np.uint8) 
    expanded = cv2.dilate(eroded, kernel, iterations=3)
    mean_values_high = []
    mean_values_low = []
    
    # Find contours
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if len(cnt) > contour_threshold]

    # Draw contours on a 3-channel image for visualization
    color_image = cv2.cvtColor(divided_normalized, cv2.COLOR_GRAY2BGR)
    thecontours = cv2.drawContours(color_image, contours, -1, (0,255,0), 2)
    
    # Iterate through each contour (particle)
    particle_id = 1
    mean_values_divided = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas_high.append(area)
        areas_low.append(area)
        
        # Create masks for the current contour
        mask_high = np.zeros_like(high_image, dtype='uint8')
        mask_low = np.zeros_like(low_image, dtype='uint8')
        mask_divided = np.zeros_like(divided_normalized)
        cv2.drawContours(mask_high, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_low, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_divided, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Extract pixels within the contour for high image (from original, not normalized)
        pixels_high = high_image[mask_high == 255]
        pixels_high = pixels_high[pixels_high != 0]
        max_high = np.max(pixels_high)
        min_high = np.min(pixels_high)
        mean_value_high = np.mean(pixels_high)
        
        # Extract pixels within the contour for low image (from original, not normalized)
        pixels_low = low_image[mask_low == 255]
        pixels_low = pixels_low[pixels_low != 0]
        max_low = np.max(pixels_low)
        min_low = np.min(pixels_low)
        mean_value_low = np.mean(pixels_low)
        
        # Extract pixels within the contour for divided image
        pixels_divided = divided_normalized[mask_divided == 255]
        pixels_divided = pixels_divided[pixels_divided != 0]
        mean_value_divided = np.mean(pixels_divided)
        max_divided = np.max(pixels_divided)
        min_divided = np.min(pixels_divided)
        if not np.isnan(mean_value_divided):
            mean_values_divided.append(mean_value_divided)
        
        # Store measurements for both channels
        if not np.isnan(mean_value_high):
            mean_values_high.append(
                {
                    "image": image_name,
                    "particle_id": particle_id,
                    "mean_value_high": mean_value_high,
                    "area_high": area,
                    "max_value_high": max_high,
                    "min_value_high": min_high,
                }
            )
        if not np.isnan(mean_value_low):
            mean_values_low.append(
                {
                    "image": image_name,
                    "particle_id": particle_id,
                    "mean_value_low": mean_value_low,
                    "area_low": area,
                    "max_value_low": max_low,
                    "min_value_low": min_low,
                }
            )
        if not np.isnan(mean_value_divided):
            mean_values_divided.append(
                {
                    "image": image_name,
                    "particle_id": particle_id,
                    "mean_value_divided": mean_value_divided,
                    "area_divided": area,
                    "max_value_divided": max_divided,
                    "min_value_divided": min_divided,
                }
            )
        
        particle_id += 1

    tifffile.imwrite(contourname, thecontours, imagej=True)
    
    # Merge high and low measurements
    df_high = pd.DataFrame(mean_values_high)
    df_low = pd.DataFrame(mean_values_low)
    df_divided = pd.DataFrame(mean_values_divided)
    mean_values_df = pd.merge(
        df_high,
        df_low,
        on=["image", "particle_id"],
        how="outer"
    )

    mean_values_df = pd.merge(
        mean_values_df,
        df_divided,
        on=["image", "particle_id"],
        how="outer"
    )

    return mean_values_df, areas_high, areas_low, mean_values_divided

def main():
    with open ("settings.yaml") as settingfile:
        settings = yaml.safe_load(settingfile)
        particle_size_threshold = settings['particle_size_threshold']
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
    particle_data = []
    mean_values_from_divided = []
    areas = []
    for image_name, image in images.items():
        aligned_high, cropped_low, divided = process_tif(image_name, image, output_path, roi_high, roi_low)
        mean_merged, areas_high, areas_low, mean_divided = calculate_mean_value_and_areas(
            divided,
            aligned_high,
            cropped_low,
            image_name,
            output_path,
            particle_size_threshold,
        )
        particle_data.append(mean_merged)
        mean_values_from_divided.extend(mean_divided)
        areas.extend(areas_high)
    print(f'Areas calculated: {len(areas)}, {plt.average(areas)}')
    
    # Save particle data (high/low measurements)
    if particle_data:
        particle_data_df = pd.concat(particle_data, ignore_index=True)
    else:
        particle_data_df = pd.DataFrame(
            columns=[
                "image",
                "particle_id",
                "mean_value_high",
                "area_high",
                "mean_value_low",
                "area_low",
            ]
        )
    particle_data_df.to_csv(f'{output_path}/particle_data.csv', index=False)
    
    # Save mean values from divided image
    with open(f'{output_path}/mean_values.csv', 'w') as f:
        for value in mean_values_from_divided:
            f.write(str(value))
            f.write(';')
    with open(f'{output_path}/areas.csv', 'w') as f:
        for value in areas:
            f.write(str(value))
            f.write(';')
                           

if __name__ == "__main__":
    main()