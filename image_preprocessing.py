#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:37:46 2024

@author: gemmagothard

overall folder structure for analysis:
    
  - raw
        - brain_ID (CBLK1234.1a_left) - saved images straight from Odyssey in nested folders
  - processed
        - Images 
          - Brain_ID (CBLK1234.1a_left)
              - brain_ID.tif (CBLK1234.1a_left_X_) - where this processing script will save them, each hemisphere will be one image, X is the slice number
        - CSVs
            - brain_ID.csv (CBLK1234.1a_left) - one CSV per brain
  - segmentation
  - annotation
  - registration
  - allen_brain_data
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
import glob
import re

from statistics import mode
from itertools import compress
from pathlib import Path
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
    parser.add_argument('brain_ID',                             help='CBLK1234_1X',                       type=str)
    args = parser.parse_args()


    # load excel spreadsheet
    excel_path = args.excel_metadata_spreadsheet
    wb = openpyxl.load_workbook(excel_path)
    slice_info = wb['slice_info']
    slice_info_df = pd.DataFrame(slice_info.values,columns=['Folder','Odyssey_file_name','Slice_sampling','Slices_omitted','Channel','Hemisphere','Slices_in_this_image','Threshold'])

    flip_image_info = wb['flip_image_info']
    

    print('loading images')
    # get data from each filename
    brain_directory = Path(args.raw_image_directory) / args.brain_ID

    # find all tif files below root directory
    tif_filepaths = [Path(f) for f in glob.glob(str(brain_directory / '*/*.TIF'))]
    tif_filenames = [f.name for f in tif_filepaths]
    
    
    order = np.argsort(tif_filenames)
    filenames_sorted = [tif_filenames[ii] for ii in order]
    filepaths_sorted = [tif_filepaths[ii] for ii in order]
    
    
    # use regular expression to search for the excitation wavelength
    pattern = r'^([0-9]*_[0-9]*)_([0-9]*)([A-z]*)-([0-9]*[A-z]*)'
    filename_ids = [int(re.search(pattern,x).group(1)) for x in filenames_sorted]
    excitation_ids = [int(re.search(pattern,x).group(2)) for x in filenames_sorted]

    unique_filenames = np.unique(filename_ids)
    
    print('segmenting images')
    slice_counter_left = 1
    slice_counter_right = 1
    for file_id in unique_filenames:

        file_bool = filename_ids==file_id
        
        # get the relevant filepaths and filenames
        filenames = list(compress(filenames_sorted,file_bool))
        filepaths = list(compress(filepaths_sorted,file_bool))
        exc_wl    = list(compress(excitation_ids,file_bool))
        
        print(filenames)

        for filen, filep in zip(filenames,filepaths):
            if int(re.search(pattern,filen).group(2))==520:
                TdTom_image = skimage.io.imread(filep)
                TdTom_image = image_uint8(TdTom_image)
            if int(re.search(pattern,filen).group(2))==488:
                GFP_image = skimage.io.imread(filep)
                GFP_image = image_uint8(GFP_image)
        
        
        # above_background_mask = TdTom_image>mode(TdTom_image.flatten())
        # mode_tdtom = mode(TdTom_image[above_background_mask])
        # mode_gfp = mode(GFP_image[above_background_mask])
       
        exposure_adjust = skimage.exposure.adjust_log(TdTom_image, gain=6, inv=False)
        
        bounding_boxes = []
        fig,ax = plt.subplots(1,figsize=(10,8))

        selectors = []
        ax.imshow(exposure_adjust)  # plot something
        ax.set_title(file_id)
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
        slice_sampling_info = slice_info_df[slice_info_df.Odyssey_file_name==filenames[0]].Slice_sampling.values[0]
        hemisphere = slice_info_df[slice_info_df.Odyssey_file_name==filenames[0]].Hemisphere.values[0]
        
        slices_in_image_left = []
        slices_in_image_right = []
        for c,b in enumerate(bounding_boxes):
            
            x1, x2, y1, y2 = [int(np.ceil(x)) for x in b]
            
            this_TdTom_brain = TdTom_image[y1:y2,x1:x2]
            this_GFP_brain = GFP_image[y1:y2,x1:x2]
                

            # save file and increment slice counter by one multiplied by slice sampling frequency
            if hemisphere=='left':
                TdTom_save_filename = str(args.brain_ID + '_s{}_' + hemisphere + '_c520_.tif').format(slice_counter_left)
                GFP_save_filename = str(args.brain_ID + '_s{}_' + hemisphere + '_c488_.tif').format(slice_counter_left)
                slices_in_image_left.append(slice_counter_left)
                
                slice_counter_left += 1*slice_sampling_info
                
                
            if hemisphere=='right':
                TdTom_save_filename = str(args.brain_ID + '_s{}_' + hemisphere + '_c520_.tif').format(slice_counter_right)
                GFP_save_filename = str(args.brain_ID + '_s{}_' + hemisphere + '_c488_.tif').format(slice_counter_right)
                slices_in_image_right.append(slice_counter_right)
                
                slice_counter_right += 1*slice_sampling_info
                
                
            
            # if the directory does not already exist, create it
            image_save_dir = os.path.join(args.image_output_directory,args.brain_ID)
            if not os.path.isdir(image_save_dir):
                os.makedirs(image_save_dir)
                
            skimage.io.imsave(os.path.join(image_save_dir,TdTom_save_filename),image_uint8(this_TdTom_brain))
            skimage.io.imsave(os.path.join(image_save_dir,GFP_save_filename),image_uint8(this_GFP_brain))

            
            
            flip_image_info.append({'A':args.brain_ID,
                                    'B':TdTom_save_filename,
                                    #'E':mode_gfp
                                    })
        
            wb.save(excel_path)
            
        for row in slice_info.iter_rows():
            if row[1].value==filenames[0]:
                row[6].value = str(slices_in_image_left)
                
        wb.save(excel_path)
                
            
            
            
            
            
            













