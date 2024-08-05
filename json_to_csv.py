# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:14:18 2024

@author: akerm
"""

import json
import csv



with open(r"C:\Users\akerm\OneDrive - Nexus365\Postdoc\Tracing_experiments\analysis_test_v2\chat_outputs\registration\brain_atlas_reg_manual.json") as json_file:
	jsondata = json.load(json_file)


import pandas as pd

df = pd.read_json(jsondata)





#%%
output_df = pd.DataFrame(columns=['Filenames','ox','oy','oz','ux','uy','uz','vx','vy','vz','width','height','nr','depths'])

output_df['Filenames'] = df.filename
output_df['width'] = df.width
output_df['height'] = df.height
output_df['nr'] = df.nr
#output_df['depths'] =

output_df['ox'] = df.anchoring.map(lambda x: x[0])
output_df['oy'] = df.anchoring.map(lambda x: x[1])
output_df['oz'] = df.anchoring.map(lambda x: x[2])

output_df['ux'] = df.anchoring.map(lambda x: x[3])
output_df['uy'] = df.anchoring.map(lambda x: x[4])
output_df['uz'] = df.anchoring.map(lambda x: x[5])

output_df['vx'] = df.anchoring.map(lambda x: x[6])
output_df['vy'] = df.anchoring.map(lambda x: x[7])
output_df['vz'] = df.anchoring.map(lambda x: x[8])



output_df.to_csv(r"C:\Users\akerm\OneDrive - Nexus365\Postdoc\Tracing_experiments\analysis_test_v2\chat_outputs\registration\brain_atlas_reg_manual.csv")







