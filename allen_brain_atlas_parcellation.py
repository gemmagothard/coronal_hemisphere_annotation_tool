#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from netgraph import InteractiveGraph, get_sugiyama_layout
from allensdk.core.reference_space_cache import ReferenceSpaceCache

# initialise figure
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['savefig.dpi'] = 300

# Load ABA reference.
rspc = ReferenceSpaceCache(
    resolution=25,
    reference_space_key='annotation/ccf_2017',
    manifest="/home/paul/src/coronal_hemisphere_annotation_tool/data/manifest.json"
)
tree = rspc.get_structure_tree(structure_graph_id=1) # ID 1 is the adult mouse structure graph
aba_id_to_name = tree.get_name_map()
aba_name_to_id = {aba_name : aba_id for aba_id, aba_name in sorted(aba_id_to_name.items())}

# BFS traversal for k levels
k = 2
root = 997
next_level = {aba_name_to_id["Cerebrum"], aba_name_to_id["Brain stem"]}
edges = [(root, node) for node in next_level]
for ii in range(k):
    this_level = next_level
    next_level = set()
    for node in this_level:
        for child in tree.child_ids([node])[0]:
            next_level.add(child)
            edges.append((node, child))

k = 2
next_level = {aba_name_to_id["Thalamus"], aba_name_to_id["Cortical plate"]}
for ii in range(k):
    this_level = next_level
    next_level = set()
    for node in this_level:
        for child in tree.child_ids([node])[0]:
            next_level.add(child)
            edges.append((node, child))

k = 1
next_level = {
    aba_name_to_id["Somatosensory areas"],
    aba_name_to_id["Somatomotor areas"],
    aba_name_to_id["Ventral group of the dorsal thalamus"],
    aba_name_to_id["Lateral group of the dorsal thalamus"],
}
for ii in range(k):
    this_level = next_level
    next_level = set()
    for node in this_level:
        for child in tree.child_ids([node])[0]:
            next_level.add(child)
            edges.append((node, child))

k = 1
next_level = {
    aba_name_to_id["Primary somatosensory area"],
}
for ii in range(k):
    this_level = next_level
    next_level = set()
    for node in this_level:
        for child in tree.child_ids([node])[0]:
            next_level.add(child)
            edges.append((node, child))

# plot graph
fig, ax = plt.subplots(figsize=(10, 5))
node_positions = get_sugiyama_layout(edges, scale=(1, 2))
node_positions = {node : (2 - y, x) for node, (x, y) in node_positions.items()}
node_labels = {node : name for node, name in tree.get_name_map().items() if node in np.unique(edges)}
plot = InteractiveGraph(edges, node_layout=node_positions, node_size=0.2, edge_width=0.1, node_labels=node_labels, node_label_offset=(0, 0.005), scale=(2,1), ax=ax)
plt.show()

# export leaf names and IDs
g = nx.DiGraph(edges)
leaves = [node for node, degree in dict(g.out_degree()).items() if degree == 0]
parcellation = {aba_id_to_name[leaf] : leaf for leaf in leaves}
with open("aba_parcellation.json", "w") as f:
    json.dump(parcellation, f)
