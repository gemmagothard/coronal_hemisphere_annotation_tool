#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:38:33 2024

@author: gemmagothard
"""


from allensdk.core.reference_space_cache import ReferenceSpaceCache


# Load ABA reference.
rspc = ReferenceSpaceCache(
    resolution=25,
    reference_space_key='annotation/ccf_2017',
    manifest="/Users/gemmagothard/Library/CloudStorage/OneDrive-Nexus365/Postdoc/Tracing_experiments/coronal_hemisphere_annotation_tool/allen_brain_data/manifest.json"
)
tree = rspc.get_structure_tree(structure_graph_id=1) # ID 1 is the adult mouse structure graph

# Convert annotation name to the ABA annotation ID
aba_id_to_name = tree.get_name_map()
aba_name_to_id = {aba_name : aba_id for aba_id, aba_name in sorted(aba_id_to_name.items())}


# Coarse-grain region assignment
ancestor_id_map = tree.get_ancestor_id_map()
parcellation = {'Prelimbic area': 972,
                'Retrosplenial area': 254,
                'Pallidum': 803,
                'Orbital area': 714,
                'Primary motor area': 985,
                'Supplemental somatosensory area': 378,
                'Infralimbic area': 44,
                'Visceral area': 677,
                'Endopiriform nucleus': 942,
                'Striatum': 477,
                'Agranular insular area': 95,
                'Secondary motor area': 993,
                'Anterior cingulate area': 31,
                'Olfactory areas': 698,
                'Primary somatosensory area': 322,
                'Gustatory areas': 1057,
                'corpus callosum': 776,
                'fiber tracts':1009,
                'Hypothalamus':1097,
                'Anterior group of the dorsal thalamus':239,
                'Claustrum':583,
                
                }

def get_ancestor(aba_id):
    for target_id in parcellation.values():
        if target_id in ancestor_id_map[aba_id]:
            return target_id
    else:
        raise ValueError(f"No target ABA ID specified for {aba_id_to_name[aba_id]} : {aba_id}!")


query_id = 500

print(aba_id_to_name[query_id])

print([aba_id_to_name[x] for x in ancestor_id_map[query_id]])

print([x for x in ancestor_id_map[query_id]])



query_name = 'Brain stem'

print(query_name)
print(aba_name_to_id[query_name])


#[print(x) for x in aba_name_to_id if query_name in x]

#[print(aba_id_to_name[aba_name_to_id==x]) for x in aba_name_to_id if query_name in x]











