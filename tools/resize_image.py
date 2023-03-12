#!/usr/bin/python
from PIL import Image
import os, sys
import pathlib
from imutils import paths
import pdb

data_path = pathlib.Path.cwd() / "data"
# Actual path to the image dataset
image_data_path = pathlib.Path(data_path) / "test_dataset" / "3"

imagePaths = list(paths.list_images(image_data_path))

for imagepath in imagePaths:
    path_parts = path_parts = imagepath.split(os.path.sep)
    filename = path_parts[-1]
    # pdb.set_trace()
    im = Image.open(imagepath)
    imResize = im.resize((224,224), Image.ANTIALIAS)
    imResize.save(os.path.join(image_data_path, filename), quality=90)

print('resizing complete')

