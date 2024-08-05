#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:37:46 2024

@author: gemmagothard

overall folder structure for analysis:
    
  - raw
        - brain_ID (CBLK1234.1a_left) - saved images straight from Odyssey
          - saved_image.tif
  - processed
        - Images 
          - Brain_ID (CBLK1234.1a_left)
              - brain_ID.tif (CBLK1234.1a_left_X_) - where this processing script will save them, each hemisphere will be one image, X is the slice number
        - CSVs
            - brain_ID.csv (CBLK1234.1a_left) - one CSV per brain
  - chat_outputs
          - segmentation
          - annotation
          - registration
          - allen_data
          - figures


This script will:
- take the raw .tif files saved from the Odyssey 
- segment them so individual slices are saved as individual images - these are saved in 'processed > images', it will make one folder per brain
- allow you to flip the images if necessary so they are in the correct orientation for Paul's alignment script 
- identify the labelled cells and saves these in an csv file ready for Paul's analysis - one csv file per brain

"""


from argparse import ArgumentParser, RawDescriptionHelpFormatter

import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.ndimage as ndimage  
import openpyxl 

from scipy.spatial import KDTree 
from skimage import measure
from matplotlib.widgets import RectangleSelector
from shapely import within
from shapely.geometry.polygon import Polygon, Point
from datetime import date
from scribble_GG_version import Scribbler


#%%  FUNCTIONS


def pad_to_square(image):
    """
    Pad a rectangular image to make it square.

    Parameters:
    - image: 2D numpy array (grayscale) or 3D numpy array (RGB)
      The input image to be padded.

    Returns:
    - padded_image: 2D or 3D numpy array
      The padded square image.
    """
    # Get the dimensions of the original image
    original_height, original_width = image.shape[:2]

    # Determine the new dimensions for the square image
    max_dim = max(original_height, original_width)

    # Create a new square image with the max dimension
    if len(image.shape) == 2:
        # Grayscale image
        padded_image = np.zeros((max_dim, max_dim), dtype=image.dtype)
    else:
        # RGB image
        padded_image = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)

    # Calculate the starting point to center the original image
    start_y = (max_dim - original_height) // 2
    start_x = (max_dim - original_width) // 2

    # Place the original image in the center of the new square image
    padded_image[start_y:start_y + original_height, start_x:start_x + original_width] = image

    return padded_image


def kLargest(arr, k):
    
    # returns the kth largest elements in an array
    sorted_list = np.argsort([len(x) for x in arr])[-k:]
    
    return sorted_list


def select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    print(f"The buttons you used were: {eclick.button} {erelease.button}")
    
    box = [*sorted([x1,x2]), *sorted([y1,y2])]   # returns list as: [xmin, xmax, ymin, ymax]
    
    bounding_boxes.append(box)
    

def toggle_selector(event):
    print('Key pressed.')
    if event.key == 't':
        for selector in selectors:
            name = type(selector).__name__
            if selector.active:
                print(f'{name} deactivated.')
                selector.set_active(False)
            else:
                print(f'{name} activated.')
                selector.set_active(True)
                
                
                
def write_excel(filename,sheetname,dataframe):
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer: 
        workBook = writer.book
        try:
            workBook.remove(workBook[sheetname])
        except:
            print("Worksheet does not exist")
        finally:
            dataframe.to_excel(writer, sheet_name=sheetname,index=False)
            writer.save()
            
            
def create_bounding_boxes(image):
    bounding_box = list()
    fig,ax = plt.subplots(1)

    selectors = []
    ax.imshow(exposure_adjust)  # plot something
    ax.set_title("Click and drag to draw a rectangle.")
    selectors.append(RectangleSelector(
        ax, select_callback,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True))

    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()
    return bounding_box


def image_uint8(image):
    return skimage.util.img_as_ubyte(image / np.max(image))





#%%


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("raw_image_directory",                  help="/path/to/raw/image/directory",   type=str)
    parser.add_argument("image_output_directory",               help="/path/to/image/output/directory", type=str)
    parser.add_argument('excel_metadata_spreadsheet',           help='/path/to/excel/spreadhsheet.xsls', type=str)
    args = parser.parse_args()


    # load excel spreadsheet
    excel_path = args.excel_metadata_spreadsheet
    wb = openpyxl.load_workbook(excel_path)
    slice_info = wb['slice_info']
    slice_info_df = pd.DataFrame(slice_info.values,columns=['Folder','Odyssey_file_name','Slice_sampling','Slices_omitted'])

    
    flip_image_info = wb['flip_image_info']
    
    print('loading images')
    brain_ID = os.path.basename(args.raw_image_directory)
    list_of_tif_files = [file for file in os.listdir(args.raw_image_directory) if file.endswith('.TIF')]
    
    filepaths = [os.path.join(args.raw_image_directory,file) for file in list_of_tif_files]
    order = np.argsort(filepaths)
    file_numbers = [filepaths[ii] for ii in order]
    filepaths = [filepaths[ii] for ii in order]
    
    
    print('segmenting images')
    slice_counter = 1
    for ii,(tif_filename, filepath) in enumerate(zip(list_of_tif_files, filepaths)):
        print(tif_filename)
        image = skimage.io.imread(filepath)
        exposure_adjust = skimage.exposure.adjust_log( image_uint8(image), gain=1, inv=False)
        
        # slice_info.append({'A':brain_ID,
        #                    'B': tif_filename})
        # wb.save(excel_path)
        
        bounding_boxes = []
        fig,ax = plt.subplots(1,figsize=(10,8))

        selectors = []
        ax.imshow(exposure_adjust)  # plot something
        ax.set_title("Click and drag to draw a rectangle.")
        selectors.append(RectangleSelector(
            ax, select_callback,
            useblit=True,
            button=[1, 3],  # disable middle button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True))

        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        
        # what is the slice sampling frequency of the current slide?
        slice_sampling_info = slice_info_df[slice_info_df.Odyssey_file_name==tif_filename].Slice_sampling.values[0]
        
        for c,b in enumerate(bounding_boxes):
            
            x1, x2, y1, y2 = [int(np.ceil(x)) for x in b]
            
            this_brain = image[y1:y2,x1:x2]
            
            #this_brain = pad_to_square(this_brain)
            #plt.imshow(image_uint8(this_brain))
            #plt.show()
            # print('flip so midline is on right side?')
            # if input().lower() == 'y': # paul's registration only works when midline is on right
            #     this_brain = np.flip(this_brain,1)
            #     print('flipping brain')
                

            # save as tiff file
            save_filename = str(brain_ID + '_{}_.tif').format(slice_counter)
            
            # if the directory does not already exist, create it
            image_save_dir = os.path.join(args.image_output_directory,brain_ID)
            if not os.path.isdir(image_save_dir):
                os.makedirs(image_save_dir)
            skimage.io.imsave(os.path.join(image_save_dir,save_filename),image_uint8(this_brain))
            
            # increment slice counter by one multiplied by slice sampling frequency
            slice_counter += 1*slice_sampling_info
            
            
            flip_image_info.append({'A':brain_ID,
                                    'B': save_filename})
        
            wb.save(excel_path)
            
            
            
            
            
            













