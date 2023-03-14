import argparse
import os
# import splitfolders 
import pdb
import os
import numpy as np
import shutil
import random

# creating train / val /test
DATA_DIR = 'data'

ORG_1_DIR = 'test_dataset_1'
ORG_2_DIR = 'test_dataset_2'
ORG_3_DIR = 'test_dataset_3'
# NEW_DIR = '/splitted/dataset'
CLASSES = ["1", "2", "3"]
IMAGE_DATASET_DIR = "data/test_dataset"

for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, ORG_1_DIR, cls))
    os.makedirs(os.path.join(DATA_DIR, ORG_2_DIR, cls))
    os.makedirs(os.path.join(DATA_DIR, ORG_3_DIR, cls))

## creating partition of the data after shuffeling
val_ratio = 0.35
test_ratio = 0.35

for cls in CLASSES:
    src = os.path.join(IMAGE_DATASET_DIR, cls) # folder to copy images from
    print(src)

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    ## here 0.75 = training ratio , (0.95-0.75) = validation ratio , (1-0.95) =  
    ##training ratio  
    # train_FileNames,test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8),int(len(allFileNames)*0.85)])
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                               int(len(allFileNames) * (1 - val_ratio)),
                                                               ])
    # train_FileNames, test_FileNames = np.split(np.array(allFileNames),
    #                                                         [int(len(allFileNames) * (1 - val_ratio)),
    #                                                         ])
    # #Converting file names from array to list

    train_FileNames = [src+'/'+ name for name in train_FileNames]
    val_FileNames = [src+'/' + name for name in val_FileNames]
    test_FileNames = [src+'/' + name for name in test_FileNames]

    print('Total images  : '+ cls + ' ' +str(len(allFileNames)))
    print('Training : '+ cls + ' '+str(len(train_FileNames)))
    print('Validation : '+ cls + ' ' +str(len(val_FileNames)))
    print('Testing : '+ cls + ' '+str(len(test_FileNames)))
    
    ## Copy pasting images to target directory

    for name in train_FileNames:
        shutil.copy(name, os.path.join(DATA_DIR, ORG_1_DIR, cls))

    for name in val_FileNames:
        shutil.copy(name, os.path.join(DATA_DIR, ORG_2_DIR, cls))

    for name in test_FileNames:
        shutil.copy(name, os.path.join(DATA_DIR, ORG_3_DIR, cls))