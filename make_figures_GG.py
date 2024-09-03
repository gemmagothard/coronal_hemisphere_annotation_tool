# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:55:47 2024

@author: akerm
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')


# if __name__ == "__main__":

#     parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

#     parser = ArgumentParser()
#     parser.add_argument("segmentation",     help="/path/to/segmentation/directory/", type=str)
#     parser.add_argument("annotation",       help="/path/to/annotation/directory/",   type=str)
#     parser.add_argument("sample_data",      help="/path/to/sample_data.csv",         type=str)
#     parser.add_argument("output_directory", help="/path/to/ouput/directory/",        type=str)
#     args = parser.parse_args()

#     output_directory = Path(args.output_directory)
#     output_directory.mkdir(exist_ok=True)

#     # load segmentation
#     segmentation_directory = Path(args.segmentation)
#     data = np.load(segmentation_directory / "segmentation_results.npz")
#     slice_images  = data["slice_images"]
#     slice_masks   = data["slice_masks"]
#     sample_masks  = data["sample_masks"]
#     slice_numbers = data["slice_numbers"]

#     df = pd.read_csv(args.sample_data)

#     # load annotation
#     data = np.load(Path(args.annotation) / "annotation_results.npz", allow_pickle=True)
#     annotations     = data["annotations"]
#     color_map_array = data["color_map"]
#     name_map_array  = data["name_map"]

#     color_map = dict(zip(color_map_array[:, 0], color_map_array[:, 1:]))
#     name_map = dict(name_map_array)





if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser = ArgumentParser()
    parser.add_argument("segmentation",     help="/path/to/segmentation/directory/", type=str)
    parser.add_argument("annotation",       help="/path/to/annotation/directory/",   type=str)
    parser.add_argument("sample_data",      help="/path/to/csv/",                    type=str)
    parser.add_argument("output_directory", help="/path/to/ouput/directory/",        type=str)
    parser.add_argument("brain_ID",         help='CBLK1234_1X',                      type=str)
    args = parser.parse_args()
    
    
    
    

# load annotation
annotation_data = np.load(,allow_pickle=True)
segmentation_data = np.load(r"C:\Users\akerm\OneDrive - Nexus365\Postdoc\Tracing_experiments\analysis_test\chat_outputs\segmentation\segmentation_results.npz",allow_pickle=True)

annotations     = annotation_data["annotations"]
color_map_array = annotation_data["color_map"]
name_map_array  = annotation_data["name_map"]

color_map = dict(zip(color_map_array[:, 0], color_map_array[:, 1:]))
name_map = dict(name_map_array)



sample_data = pd.read_csv(r"C:\Users\akerm\OneDrive - Nexus365\Postdoc\Tracing_experiments\analysis_test\processed\csv\CBLK1234_1C_left.csv")

# list of areas which should not contain cells (these cells need to be removed)
fissures_and_tracts = [x for x in name_map.values() if np.any(['tract' in x,'fissure' in x,'ventricle' in x, 'Caudo' in x,'corpus' in x,'commissure' in x])]
fissures_and_tracts = [x for x in fissures_and_tracts if 'nucleus' not in x if 'Nucleus' not in x]

sample_data = sample_data[np.invert(sample_data['annotation_name'].isin(fissures_and_tracts))]


sample_data.to_csv(r'C:\Users\akerm\OneDrive - Nexus365\Postdoc\Tracing_experiments\analysis_test\processed\csv\CBLK1234_1C_left.csv')

#%%

fig,ax = plt.subplots(1,1,figsize=(20,20))

plt.hist(sample_data.annotation_name,orientation=u'horizontal')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.ylim(0,40)







