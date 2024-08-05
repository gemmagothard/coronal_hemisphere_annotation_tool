# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:47:23 2024

@author: akerm
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
from shapely import within
from shapely.geometry.polygon import Polygon, Point
from datetime import date
from scribble_GG_version import Scribbler


#%%  FUNCTIONS


def find_cells(image,file):
    
    #determine threshold based on image statistics
    
    # find the longest contour in this image
    contours = measure.find_contours(image.astype(float),np.median(image)*3)
    longest_contour_idx = np.argmax([len(x) for x in contours])
    longest_contour = contours[longest_contour_idx]
    
    
    # Create an empty image to store the masked array
    r_mask = np.zeros_like(image, dtype='bool')
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    r_mask[np.round(longest_contour[:, 0]).astype('int'), np.round(longest_contour[:, 1]).astype('int')] = 1
    # Fill in the hole created by the contour boundary
    r_mask = ndimage.binary_fill_holes(r_mask)
    
    th = np.median(image[r_mask])*0.1

    #DOG - difference of gaussians 
    blobs = skimage.feature.blob_dog(image.astype(float), min_sigma=1, max_sigma=2, threshold=th, overlap=0.9)
    
    # find distances from blobs to the contours
    tree = KDTree(longest_contour)
    distance, blob_idx = tree.query(blobs[:,:2],1)

    # remove the blobs that are too close to the contour edges
    blobs_edges_removed = blobs[distance>18]
    
    # remove third column - this is standard deviation related to blob detection
    cells = np.delete(blobs_edges_removed,2,1)
    
    # remove any cells that are found outside of the slice - these are obvious mistakes
    cells_in_slice_mask = [within(Point((b[0],b[1])),Polygon(longest_contour)) for b in cells]
    cells = cells[cells_in_slice_mask]
    
    
    
    # manually remove false positive cells
    scribbler = Scribbler(image,cells,title=file)
    scribble_masks = scribbler.get_scribble_masks()

    cell_mask = np.zeros(image.shape)
    cell_mask[cells[:,0].astype(int), cells[:,1].astype(int)] = 1
    
    # these are the cells to keep
    if np.sum(scribble_masks['1'])>0:
        scribble_cell_mask = np.logical_not(scribble_masks['1']) * cell_mask
        cells_fr = np.argwhere(scribble_cell_mask)
    else:
        cells_fr = cells

    plt.imshow(image)
    plt.scatter(cells[:,1],cells[:,0],facecolor='red',s=4)
    plt.scatter(cells_fr[:,1],cells_fr[:,0],facecolor='green',s=4)
    plt.title(file)
    
    plt.show()
    
    return cells_fr
    


#%%





if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("csv_output_directory",                 help="/path/to/csv/output/directory",   type=str)
    parser.add_argument("image_processed_directory",            help="/path/to/image/output/directory", type=str)
    parser.add_argument('excel_metadata_spreadsheet',           help='/path/to/excel/spreadsheet.xsls', type=str)
    parser.add_argument('--slice_thickness',                    help='slice thickness in microns',      type=float)
    args = parser.parse_args()

    brain_ID = os.path.basename(args.image_processed_directory)
            
    
    # load excel spreadsheet
    excel_path = args.excel_metadata_spreadsheet
    wb = openpyxl.load_workbook(excel_path)
    flip_image_info = wb['flip_image_info']
    flip_image_info_df = pd.DataFrame(flip_image_info.values,columns=['Folder','filename','flip_image'])

    
    #  find cells, save results in csv
    print('finding cells')
    image_filepath = args.image_processed_directory
    
    # get all the filenames that end with tif
    file_list = [file for file in os.listdir(image_filepath) if file.endswith('.tif')]
    print(file_list)
    data_for_csv = []
    for i,file  in enumerate(file_list):
        
        image = skimage.io.imread(os.path.join(image_filepath,file))
        
        # flip this slice?
        if flip_image_info_df[flip_image_info_df.filename==file].flip_image.values[0] == 'Y':
                image = np.flip(image,1)
                skimage.io.imsave(os.path.join(image_filepath,file),image)
            
        
        slice_number = int(file.split('_')[-2:-1][0])

        cells_detected = find_cells(image,file)
        
        data_for_csv.append({'image_index':i,
                        'date':date.today(),
                        'slice_number':slice_number,
                        'brain_ID':brain_ID,
                       'image_row':cells_detected[:,0],
                       'image_col':cells_detected[:,1],
                       'image_z': slice_number * args.slice_thickness,
                       })
    
    slice_df = pd.DataFrame(data_for_csv)
    
    exploded_df = slice_df.explode(['image_row','image_col'])
    exploded_df['sample_id']=np.arange(0,len(exploded_df))
    
    csv_name = str(brain_ID + '.csv')
    
    if not os.path.isdir(args.csv_output_directory):
        os.makedirs(args.csv_output_directory)
    exploded_df.to_csv(os.path.join(args.csv_output_directory,csv_name))








