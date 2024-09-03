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
import glob 
import re

from statistics import mode
from itertools import compress
from pathlib import Path
from scipy.spatial import KDTree 
from skimage import measure
from shapely import within
from shapely.geometry.polygon import Polygon, Point
from datetime import date
from scribble_GG_version import Scribbler
from skimage.filters import threshold_otsu

#%%  FUNCTIONS



def find_cells(gfp,tdtom,file):

    
    background_of_slice = np.argwhere(np.log(gfp+1)>0)
    mask_of_slice = np.zeros(shape=gfp.shape)
    mask_of_slice[background_of_slice[:,0],background_of_slice[:,1]] = 1
    
    
    tdtom_cells = skimage.feature.blob_dog(tdtom.astype(float), min_sigma=1, max_sigma=4, threshold_rel=0.06, overlap=0.9)[:,:2]
    gfp_cells = skimage.feature.blob_dog(gfp.astype(float), min_sigma=1, max_sigma=4, threshold_rel=0.06, overlap=0.9)[:,:2]


    # find the longest contour in this image
    contours = measure.find_contours(tdtom,5)
    longest_contour_idx = np.argmax([len(x) for x in contours])
    longest_contour = contours[longest_contour_idx]
    
    # find distances from blobs to the contours
    tree = KDTree(longest_contour)
    distance, blob_idx = tree.query(tdtom_cells,1)
    # remove the blobs that are too close to the contour edges
    tdtom_cells = tdtom_cells[distance>18]
    
    # remove any cells that are found outside of the slice - these are obvious mistakes
    cells_in_slice_mask = [within(Point((b[0],b[1])),Polygon(longest_contour)) for b in tdtom_cells]
    tdtom_cells = tdtom_cells[cells_in_slice_mask]
    
    # find the nearest gfp+ cell to each tdtom cell
    tree = KDTree(gfp_cells[:,:2])
    nearest_dist, nearest_ind = tree.query(tdtom_cells[:,:2], k=1)

    # if the distance is <2 pixels, count as a gfp+,tdtom+ cell (starter cell) - otherwise count as just input cell (tdtom+gfp-)
    starter_cells = tdtom_cells[nearest_dist<=2]
    input_cells = tdtom_cells[nearest_dist>2]
    
    # manually remove false positive cells
    # click down to scribble on image
    # 'r' removes scribble
    # 'z' to disconnect scribble in order to zoom 
    scribbler = Scribbler(tdtom,input_cells,starter_cells,title=file)
    scribble_masks = scribbler.get_scribble_masks()

    starter_cell_mask = np.zeros(tdtom.shape)
    starter_cell_mask[starter_cells[:,0].astype(int), starter_cells[:,1].astype(int)] = 1
    
    input_cell_mask = np.zeros(tdtom.shape)
    input_cell_mask[input_cells[:,0].astype(int), input_cells[:,1].astype(int)] = 1
    
    # these are the cells to keep
    if np.sum(scribble_masks['1'])>0:
        input_scribble_cell_mask = np.logical_not(scribble_masks['1']) * input_cell_mask
        input_cells = np.argwhere(input_scribble_cell_mask)
        
        starter_scribble_cell_mask = np.logical_not(scribble_masks['1']) * starter_cell_mask
        starter_cells = np.argwhere(starter_scribble_cell_mask)


    plt.imshow(tdtom)
    plt.scatter(input_cells[:,1],input_cells[:,0],facecolor='red',s=1)
    plt.scatter(starter_cells[:,1],starter_cells[:,0],facecolor='orange',s=1)
    #plt.scatter(cells_fr[:,1],cells_fr[:,0],facecolor='green',s=1)
    plt.title(file)
    
    plt.show()
    
    return input_cells, starter_cells
    


#%%





if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("csv_output_directory",                 help="/path/to/csv/output/directory",   type=str)
    parser.add_argument("image_processed_directory",            help="/path/to/image/output/directory", type=str)
    parser.add_argument('excel_metadata_spreadsheet',           help='/path/to/excel/spreadsheet.xsls', type=str)
    parser.add_argument('--slice_thickness',                    help='slice thickness in microns',      type=float)
    parser.add_argument('brain_ID',                             help='CBLK1234_1X',                       type=str)
    args = parser.parse_args()


    # load excel spreadsheet
    wb = openpyxl.load_workbook(args.excel_metadata_spreadsheet)
    flip_image_info = wb['flip_image_info']
    flip_image_info_df = pd.DataFrame(flip_image_info.values,columns=['Folder','filename','flip_image'])

    
    #  find cells, save results in csv
    print('finding cells')
    # get data from each filename
    brain_directory = Path(args.image_processed_directory) / args.brain_ID

    # find all tif files below root directory
    tif_filepaths = [Path(f) for f in glob.glob(str(brain_directory / '*.tif'))]
    tif_filenames = [f.name for f in tif_filepaths]

    slice_numbers = [int(re.search(r'^([A-z]*[0-9]*_[0-9]*[A-z]*_[A-z]*)([0-9]*)_[A-z]*_[A-z]*([0-9]*)_', x)[2]) for x in tif_filenames]
    order = np.argsort(slice_numbers)
    
    filenames_sorted = [tif_filenames[ii] for ii in order]
    filepaths_sorted = [tif_filepaths[ii] for ii in order]
    
    
    # use regular expression to search for the excitation wavelength
    pattern = r'^([A-z]*[0-9]*_[0-9]*[A-z]*_[A-z]*[0-9]*)_[A-z]*_[A-z]*([0-9]*)_'
    filename_ids = [re.search(pattern,x).group(1) for x in filenames_sorted]
    excitation_ids = [int(re.search(pattern,x).group(2)) for x in filenames_sorted]
    
    unique_filenames = np.unique(filename_ids)
    

    data_for_csv = []
    for i,file_id  in enumerate(unique_filenames):
        
        
        file_bool = [np.where(x==file_id,True,False) for x in filename_ids]

        # get the relevant filepaths and filenames
        filenames = list(compress(filenames_sorted,file_bool))
        filepaths = list(compress(filepaths_sorted,file_bool))
        exc_wl    = list(compress(excitation_ids,file_bool))
        

        for filen, filep in zip(filenames,filepaths):
            if int(re.search(pattern,filen).group(2))==520:
                tdtom_filep = filep
                TdTom_image = skimage.io.imread(filep)
                tdtom_filen = filen
            if int(re.search(pattern,filen).group(2))==488:
                GFP_image = skimage.io.imread(filep)
                gfp_filep = filep
        

        # flip this slice?
        if flip_image_info_df[flip_image_info_df.filename==tdtom_filen].flip_image.values[0] == 'Y':
                TdTom_image = np.flip(TdTom_image,1)
                GFP_image = np.flip(GFP_image,1)
                
                skimage.io.imsave(os.path.join(filepaths[0],tdtom_filep),TdTom_image)
                skimage.io.imsave(os.path.join(filepaths[0],gfp_filep),GFP_image)
        
        
        slice_no_pattern = r'^([A-z]*[0-9]*_[0-9]*[A-z]*_[A-z]*)([0-9]*)'
        slice_number = int(re.search(slice_no_pattern,file_id).group(2))
        
        input_cells, starter_cells = find_cells(GFP_image,TdTom_image,file_id)
        

        data_for_csv.append({'image_index':i,
                        'date':date.today(),
                        'slice_number':slice_number,
                        'brain_ID':args.brain_ID,
                        'filename':filen,  # this will contain info on the hemisphere
                       'image_row':np.concatenate([input_cells[:,0],starter_cells[:,0]]),
                       'image_col':np.concatenate([input_cells[:,1],starter_cells[:,1]]),
                       'cell_idx':['input'] * len(input_cells) + ['starter'] * len(starter_cells),
                       'image_z': slice_number * args.slice_thickness,
                       })
    
    slice_df = pd.DataFrame(data_for_csv)
    
    exploded_df = slice_df.explode(['image_row','image_col','cell_idx'])
    exploded_df['sample_id']=np.arange(0,len(exploded_df))
    
    csv_name = str(args.brain_ID + '.csv')
    
    if not os.path.isdir(args.csv_output_directory):
        os.makedirs(args.csv_output_directory)
    exploded_df.to_csv(os.path.join(args.csv_output_directory,csv_name))








