#!/usr/bin/env python
'''
File        :   create_images_dataset.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   For large sets of images, it is advised to submit a job and run this script 
                over the Jupyter Notebook
'''
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import os
import openslide
from PIL import Image, ImageOps

ORGANIZED_BY_CASE_PATH = "/users/anair27/data/TCGA_Data/project_LUAD/data_by_cases"
DESTINATION_DATA_PATH = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/PRCSD_images_data.csv"
IMAGE_PATH = "/users/anair27/data/TCGA_Data/project_LUAD/data_processed/imaging_data_updated"

cases = os.listdir(ORGANIZED_BY_CASE_PATH)
def get_FFPE_images(case):
    img_files = os.listdir(os.path.join(ORGANIZED_BY_CASE_PATH, case, "images"))
    for f in img_files:
        if (f.split('.')[0][-3:-1] == 'DX'):
            return os.path.join(ORGANIZED_BY_CASE_PATH, case, 'images', f)
    print("No DX found for case",case)
    return None

j = 0
case_and_image = {}
for case in cases:
    n = get_FFPE_images(case)
    if n is not None:
        case_and_image[case] = n
        j+=1
print(f"{j} cases out of {len(cases)} have valid images")

total_cases = len(case_and_image)
orig_dims = []
for i,c in enumerate(case_and_image):
    svs = openslide.OpenSlide(case_and_image[c])
    orig_dims.append(svs.level_dimensions[0])
    print(f"{i}/{total_cases}")
flipped_data = [(y, x) if x < y else (x, y) for x, y in orig_dims]
aspect_ratio = [x/y for x, y in flipped_data]

med_aspect_ratio = round(np.median(aspect_ratio), 4)
print(f"Median aspect ratio: {med_aspect_ratio}")

aspect_ratio = med_aspect_ratio
# SPECIFY THE OUTPUT DIMENSION FOR HEIGHT
target_h = 300
target_w = (int)(aspect_ratio * target_h)
print("Dimensions of output image", target_w, target_h)

for i,c in enumerate(case_and_image):
    # open slide
    svs = openslide.OpenSlide(case_and_image[c])
    # get a thumbnail from svs (ensure this constant is strictly less than all dims across images)
    thumbnail = svs.get_thumbnail((1000, 1000))
    dim_tn_x, dim_tn_y = thumbnail.size
    # rotate images that are vertical
    if dim_tn_y > dim_tn_x:
        thumbnail = thumbnail.transpose(method=Image.Transpose.ROTATE_90)
    # resize image and use padding to ensure that target size is guaranteed
    resized_image = ImageOps.pad(thumbnail.resize((target_w, target_h)), size=(target_w, target_h))
    output_path = os.path.join(IMAGE_PATH, f"{c}.jpeg")
    assert resized_image.size[0] == target_w
    assert resized_image.size[1] == target_h
    resized_image.save(output_path)

all_images_data = pd.DataFrame.from_dict(case_and_image, orient='index', columns=['image_path']).reset_index(\
                                                      ).rename(columns={'index':'case_id'})
all_images_data.to_csv(DESTINATION_DATA_PATH)