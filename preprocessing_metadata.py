#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:18:10 2024

@author: gemmagothard
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import re
import glob
import openpyxl
import numpy as np

from pathlib import Path


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("raw_image_directory",                  help="/path/to/raw/image/directory",   type=str)
    parser.add_argument('excel_path',                           help='/path/to/excel/spreadhsheet.xsls', type=str)
    parser.add_argument('brain_ID',                            help='brain ID', type=str)
    args = parser.parse_args()


    
    # get data from each filename
    brain_directory = Path(args.raw_image_directory) / args.brain_ID
    
    # find all tif files below root directory
    tif_filepaths = [Path(f) for f in glob.glob(str(brain_directory / '*/*.TIF'))]
    tif_filenames = [f.name for f in tif_filepaths]
    
    
    # use regular expression to search for the excitation wavelength
    pattern = r'^([0-9]*_[0-9]*)_([0-9]*)([A-z]*)-([0-9]*[A-z]*)'
    excitation_wavelengths = [int(re.search(pattern,x).group(2)) for x in tif_filenames]
                
    
    # save to metadata spreadsheet
    wb = openpyxl.load_workbook(args.excel_path)
    slice_info = wb['slice_info']
    
    order = np.argsort(tif_filenames)
    files_sorted = [tif_filenames[ii] for ii in order]
    wavelengths_sorted = [excitation_wavelengths[ii] for ii in order]
    
    for file, wavelength in zip(files_sorted, wavelengths_sorted):
        
        slice_info.append({'A':args.brain_ID,
                          'B':file,
                          'E':wavelength})
    
    wb.save(args.excel_path)
    
