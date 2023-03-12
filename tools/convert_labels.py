import os 
import numpy as np 
import pathlib
import pdb
import cv2
from imutils import paths
import pandas as pd
import argparse

# Use arg parser to pass the folder directory to convert into annotations labels
ap = argparse.ArgumentParser()

ap.add_argument(
    "-f",
    "--folder",
    required=True,
    type=str,
    help="Folders to use to convert the labels"
)

ap.add_argument(
    "-o",
    "--output",
    required=True,
    type=str,
    help="Folders to save the labels"
)

args = vars(ap.parse_args())
FOLDER = args['folder']
OUTPUT = args['output']

# Root directory of the datasets
data_path = pathlib.Path.cwd() / "data"

# Actual path to the image dataset
image_data_path = pathlib.Path(data_path) / f"{FOLDER}"

# List all the images within the image path
imagePaths = list(paths.list_images(image_data_path))

file_name_ls = []
label_ls = []
for img in imagePaths:
    path_parts = path_parts = img.split(os.path.sep)
    label = path_parts[-2]
    file_name = path_parts[-1]
    file_name_ls.append(file_name)
    label_ls.append(label)

image_dict = {
            'Image': file_name_ls,
            'label': label_ls
          }
# Conver from dataframe to csv
df = pd.DataFrame(image_dict)
df.to_csv(f'data/{OUTPUT}.csv', index=False) # Use Tab to seperate data




