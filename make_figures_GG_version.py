"""
Create an output PDF that displays each segmented slice,
the corresponding Allen Brain Atlas annnotation,
and the isolated samples including their barcode.

Example
-------
python code/05_make_figures.py test/segmentation/ test/annotation/ test/sample_data.csv test/figures/

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from matplotlib.backends import backend_pdf
from matplotlib.colors import to_hex
from skimage.measure import label, find_contours
from shapely.geometry import Polygon, LineString
from shapely.ops import polylabel
from itertools import combinations
from scipy.optimize import minimize, NonlinearConstraint
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from collections import Counter


def get_well_spaced_angles(angles, minimum_delta_angle=np.pi/9):

    def get_angle_difference(a, b):
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    def cost_function(new, old):
        return np.sum(get_angle_difference(new, old))

    def constraint_function(angles):
        a, b = np.array(list(combinations(angles, 2))).T
        return get_angle_difference(a, b)

    nonlinear_constraint = NonlinearConstraint(constraint_function, lb=minimum_delta_angle, ub=np.pi + 1e-3)
    res = minimize(lambda x: cost_function(x, angles), angles, constraints=[nonlinear_constraint])
    return res.x


def get_well_spaced_vectors(vectors, minimum_delta_angle=2*np.pi/36):
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = get_well_spaced_angles(angles, minimum_delta_angle)
    return np.c_[np.cos(angles), np.sin(angles)]





if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser = ArgumentParser()
    parser.add_argument("segmentation",     help="/path/to/segmentation/directory/", type=str)
    parser.add_argument("annotation",       help="/path/to/annotation/directory/",   type=str)
    parser.add_argument("sample_data",      help="/path/to/csv/",                    type=str)
    parser.add_argument("output_directory", help="/path/to/ouput/directory/",        type=str)
    parser.add_argument('allen_brain_path', help='/path/to/allen_brain/folder/',  type=str)
    parser.add_argument("brain_ID",         help='CBLK1234_1X',                      type=str)
    args = parser.parse_args()

    output_directory = Path(args.output_directory) / args.brain_ID
    output_directory.mkdir(exist_ok=True)

    # load segmentation
    segmentation_directory = Path(args.segmentation) / args.brain_ID
    data = np.load(segmentation_directory / "segmentation_results.npz")
    slice_images  = data["slice_images"]
    slice_masks   = data["slice_masks"]
    sample_masks  = data["sample_masks"]
    slice_numbers = data["slice_numbers"]

    csv_path = glob.glob(str(args.sample_data + args.brain_ID + '*.csv'))[0]
    df = pd.read_csv(csv_path)

    # load annotation
    data = np.load(Path(args.annotation) / args.brain_ID / str(args.brain_ID + "_annotation_results.npz"), allow_pickle=True)
    annotations     = data["annotations"]
    color_map_array = data["color_map"]
    name_map_array  = data["name_map"]
    
    
    # Load ABA reference.
    rspc = ReferenceSpaceCache(
        resolution=25,
        reference_space_key='annotation/ccf_2017',
        manifest=Path(args.allen_brain_path) / 'manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1) # ID 1 is the adult mouse structure graph
    
    # Convert annotation name to the ABA annotation ID
    aba_id_to_name = tree.get_name_map()
    aba_name_to_id = {aba_name : aba_id for aba_id, aba_name in sorted(aba_id_to_name.items())}
    df["annotation_id"] = df["annotation_name"].apply(lambda x: aba_name_to_id[x])
    
    print(df[df['annotation_id']==709])
    

    # Coarse-grain region assignment
    ancestor_id_map = tree.get_ancestor_id_map()
    with open(Path(args.allen_brain_path) / 'aba_parcellation.json' ) as f:
        parcellation = json.load(f)
        
    parcellation.update({'fiber tracts':1009,
                         'Olfactory areas': 698,
                         'Brain stem': 343  ,
                         'ventricular systems': 73,
                         'root':997,})

    def get_ancestor(aba_id):
        for target_id in parcellation.values():
            if target_id in ancestor_id_map[aba_id]:
                return target_id
        else:
            raise ValueError(f"No target ABA ID specified for {aba_id_to_name[aba_id]} : {aba_id}!")

    df["simplified_annotation_id"] = df["annotation_id"].apply(get_ancestor)
    df["simplified_annotation_name"] = df["simplified_annotation_id"].apply(lambda x : aba_id_to_name[x])
    
    print(df[df['annotation_name']=='root'])

    # --------------------------------------------------------------------------------
    
    # plot overall histogram for each brain
    for ii in np.unique(df.brain_ID):
        
        this_brain = df[df.brain_ID==ii]
        
        these_input_cells_left = this_brain[np.logical_and(this_brain.cell_idx=='input',df['filename'].str.contains('left'))]
        these_input_cells_right = this_brain[np.logical_and(this_brain.cell_idx=='input',df['filename'].str.contains('right'))]
        
        these_starter_cells = this_brain[this_brain.cell_idx=='starter']  # these will only be on the left hemisphere anyway
        
        
        fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
        
        counts_input_left = Counter(these_input_cells_left.simplified_annotation_name.values)
        df1 = pd.DataFrame.from_dict(counts_input_left, orient='index',columns=['Ipsilateral inputs'])
        
        counts_input_right = Counter(these_input_cells_right.simplified_annotation_name.values)
        df2 = pd.DataFrame.from_dict(counts_input_right, orient='index',columns=['Contralateral inputs'])
        
        counts_starter = Counter(these_starter_cells.simplified_annotation_name.values)
        df3 = pd.DataFrame.from_dict(counts_starter, orient='index',columns=['Starters cells'])
        
        appended_df = df1.join([df2,df3]).sort_values(by=['Ipsilateral inputs'])
        
        print(appended_df)
        
        appended_df.plot(kind='barh',ax=ax,color=['red','green','blue'])

        ax.set_xlabel("Cell counts")
        
        plt.tight_layout()
        ax.set_title(args.brain_ID)
        plt.show()
        fig.savefig(output_directory / str(args.brain_ID + '_histogram.svg'),bbox_inches='tight')
        
        appended_df.to_csv(output_directory / str(args.brain_ID + '_histogram.csv'))
        

    color_map = dict(zip(color_map_array[:, 0], color_map_array[:, 1:]))
    name_map = dict(name_map_array)

    # determine image center
    total_slices, total_rows, total_cols = slice_images.shape
    xc = total_cols / 2
    yc = total_rows / 2

    # --------------------------------------------------------------------------------

    print("Plotting slices & annotations...")
    date, = df["date"].unique()
    date = date.replace("-", "_")

    with backend_pdf.PdfPages(output_directory / f"individual_slices_{date}.pdf") as pdf:

        # for ii in range(total_slices):
        for ii in range(total_slices):
            print(f"{ii+1} / {total_slices}")
            slice_number = slice_numbers[ii]
            slice_image  = slice_images[ii]
            slice_mask   = slice_masks[ii]
            annotation   = annotations[ii]
            sample_mask  = sample_masks[ii]

            fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 10))
            for ax in axes:
                ax.axis("off")
                ax.set_aspect("equal")
            axes[0].imshow(slice_image, cmap="gray")

            # plot slice contour
            contours = find_contours(slice_mask)

            for contour in contours:
                x = contour[:, 1]
                y = contour[:, 0]
                axes[-1].plot(x, y, "#677e8c", linewidth=1.0)

            # get the contour around the outside of the slice
            contour = sorted(contours, key = lambda x : len(x))[-1]
            slice_contour = Polygon(np.c_[contour[:, 1], contour[:, 0]])

            # plot region contours
            for annotation_id in np.unique(annotation):
                if annotation_id > 0: # 0 is background
                    region_mask = annotation == annotation_id
                    for contour in find_contours(region_mask):
                        x = contour[:, 1]
                        y = contour[:, 0]
                        axes[-1].plot(x, y, color=to_hex(color_map[annotation_id]/255), linewidth=0.25)

            subset = df[df["slice_number"] == slice_number]

            if len(subset) > 0:

                # compute slice radius, i.e. maximum distance from center
                slice_radius = 0
                center = np.array([yc, xc])
                for contour in contours:
                    delta = contour - center[np.newaxis, :]
                    distance = np.linalg.norm(delta, axis=1)
                    slice_radius = max(slice_radius, np.max(distance))

                # plot sample locations
                for _, row in subset.iterrows():
                    x = row["segmentation_col"]
                    y = row["segmentation_row"]
                    
                    if row['cell_idx'] == 'input':
                        axes[-1].plot(x, y, linestyle="", marker='o', markersize=1, color='red')
                        
                    if row['cell_idx'] == 'starter':
                        axes[-1].plot(x, y, linestyle="", marker='o', markersize=1, color='blue')
                        
                    if row['annotation_name']=='root':
                        axes[-1].plot(x, y, linestyle="", marker='o', markersize=1, color='green')

                # plot regions that have a sample in them
                region_coordinates = []
                region_labels = []
                region_colors = []
                for annotation_id in np.unique(subset["annotation_id"]):
                    region_mask = annotation == annotation_id
                    region_mask[:, :int(xc) + 1] = False
                    
                    contours = find_contours(region_mask)
                    
                    # The contour finding algorithm sometimes identifies
                    # spurious contours around the corners of the region.
                    # We hence only plot the largest contour.
                    if len(contours)>0:
                        contour = sorted(contours, key = lambda x : len(x))[-1]
                        patch = plt.Polygon(np.c_[contour[:, 1], contour[:, 0]], color=to_hex(color_map[annotation_id]/255))
                        axes[-1].add_patch(patch)
                        polygon = Polygon(np.c_[contour[:, 1], contour[:, 0]])
                        poi = polylabel(polygon, tolerance=0.1)
                        x, y = poi.x, poi.y
                        region_coordinates.append((x, y))
                        region_labels.append(name_map[int(annotation_id)])
                
                    
            fig.savefig(output_directory / f"slice_{slice_number:03d}.svg")
            pdf.savefig(fig)
            plt.close(fig)

    # --------------------------------------------------------------------------------

    print("Plotting slices aligned in 3D...")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis("off")

    for ii in range(total_slices):
        print(f"{ii+1} / {total_slices}")
        slice_number = slice_numbers[ii]
        slice_image  = slice_images[ii]
        slice_mask   = slice_masks[ii]
        annotation   = annotations[ii]
        sample_mask  = sample_masks[ii]

        # plot slice contour
        contours = find_contours(slice_mask, 0.5)
        for contour in contours:
            x = contour[:, 1]
            y = contour[:, 0]
            z = -slice_number * np.ones_like(x)
            ax.plot(z, x, -y, "#677e8c", linewidth=1.0)

        # compute slice radius
        slice_radius = 0
        center = np.array([yc, xc])
        for contour in contours:
            delta = contour - center[np.newaxis, :]
            distance = np.linalg.norm(delta, axis=1)
            slice_radius = max(slice_radius, np.max(distance))

        # plot and label samples
        for _, row in df[df["slice_number"] == slice_number].iterrows():
            x = row["segmentation_col"]
            y = row["segmentation_row"]
            xyz = np.array([-slice_number, x, -y])
            
            if row['cell_idx'] == 'input':
                ax.plot(*xyz, linestyle="", marker='.',markersize=1, color='red')
            if row['cell_idx'] == 'starter':
                ax.plot(*xyz, linestyle="", marker='.',markersize=1, color='blue')

    # # label clones
    # for barcode in np.unique(df["barcode"]):
    #     clone = df[df["barcode"] == barcode]
    #     if len(clone) > 1:
    #         Z = -clone["slice_number"]
    #         X = clone["segmentation_col"]
    #         Y = clone["segmentation_row"]
    #         zt = Z.mean()
    #         xt = xc + 2 * (X.mean() - xc)
    #         yt = yc + 2 * (Y.mean() - yc)
    #         ax.text(zt, xt, -yt, barcode + " ", ha="right")
    #         for zz, xx, yy in zip(Z, X, Y):
    #             ax.plot([zt, zz], [xt, xx], [-yt, -yy], linewidth=0.5, color="#677e8c")

    print("Select the desired view. The figure will be saved on closing.")
    plt.show()
    fig.savefig(output_directory / "reconstruction_in_3d.pdf")
    fig.savefig(output_directory / "reconstruction_in_3d.svg")
    
    
    
    
    
    
    
    
