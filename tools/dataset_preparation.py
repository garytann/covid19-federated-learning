#!/usr/bin/python

import os
import random
import shutil
import numpy as np 
import pdb
import csv
from torch.utils.data import DataLoader
import importlib.util

# Specify the path to the module
module_path = './covidLUS/load_dataset.py'

# Load the module from the specified path
spec = importlib.util.spec_from_file_location('module_name', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Import the class from the module
CovidultrasoundClass = getattr(module, 'CovidUltrasoundDataset')

# Now you can use the class in your script
# covid_ultra_sound_dataset = CovidultrasoundClass()
# instance.some_method()

# Convert the labels in csv file
def convert_labels(src : str, dst  : str):
    dataset_folder = src
    csv_file = dst

    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'label'])  # Write the header row
        
        # Iterate over the classes
        for class_folder in os.listdir(dataset_folder):
            class_path = os.path.join(dataset_folder, class_folder)
            
            # Iterate over the images in the class folder
            for image_file in os.listdir(class_path):
                # image_path = os.path.join(class_path, image_file)
                class_label = class_folder  # Use the class folder name as the class label
                
                # Write the image path and class label to the CSV file
                writer.writerow([image_file, class_label])
    
    return csv_file

# Split the data into train and test
def train_test_split(input_folder : str, n : int):

    source_folder = os.path.join('data/fl_dataset', input_folder)
    substring = source_folder.split("/")[-1]
    org_id = substring.replace('cross_validation_', '')

    destination_folder = 'data/fl_dataset'

    # Create test_dataset folder
    test_folder = os.path.join(destination_folder, f'test_dataset_{org_id}')
    # os.makedirs(test_folder, exist_ok=True)

    # Move all files from split_0 to test_dataset folder
    # for _ in range(1, n+1):
    cross_validation_folder = os.path.join(source_folder, 'split0')
    for class_folder in os.listdir(cross_validation_folder):
        class_source = os.path.join(cross_validation_folder, class_folder)
        class_destination = os.path.join(test_folder, class_folder)
        # pdb.set_trace()
        shutil.copytree(class_source, class_destination, dirs_exist_ok=True)
    
    # Convert test_dataset_org_n into file labels
    test_label_file = convert_labels(test_folder, os.path.join(destination_folder, f'test_dataset_labels_{org_id}.csv'))

    test_covid_dataset = CovidultrasoundClass(
                                        annotations_file = test_label_file,
                                        img_dir = test_folder
                                        )
    
    test_dataloader = DataLoader(
                    test_covid_dataset, 
                    batch_size=len(test_covid_dataset), 
                    shuffle=True
                )
    
    # Iterate through the data loader and convert the features and labels to numpy 
    test_features, test_labels = next(iter(test_dataloader))
    test_dataset_numpy = test_features.numpy()
    test_labels_numpy = test_labels.numpy()

    # Save the test numpy 
    test_dir = os.path.join(destination_folder,f"client_{org_id}", "test")
    os.makedirs(test_dir, exist_ok=True)
    np.save(os.path.join(test_dir, 'test_images.npy'), test_dataset_numpy)
    np.save(os.path.join(test_dir, 'test_labels.npy'), test_labels_numpy)

    # Create train_dataset folder
    train_folder = os.path.join(destination_folder, f'train_dataset_{org_id}')

    # Move files from split_1 to split_4 to train_dataset folder
    for _ in range(1, n+1):
        for j in range(1, 5):
            cross_validation_folder = os.path.join(source_folder, f'split{j}')
            for class_folder in os.listdir(cross_validation_folder):
                class_source = os.path.join(cross_validation_folder, class_folder)
                class_destination = os.path.join(train_folder, class_folder)
                # pdb.set_trace()
                shutil.copytree(class_source, class_destination, dirs_exist_ok=True)

    train_label_file = convert_labels(train_folder, os.path.join(destination_folder, f'train_dataset_labels_{org_id}.csv'))

    train_covid_dataset = CovidultrasoundClass(
                                        annotations_file = train_label_file,
                                        img_dir = train_folder
                                        )

    train_dataloader = DataLoader(
                    train_covid_dataset, 
                    batch_size=len(train_covid_dataset), 
                    shuffle=True
                )
    
    # Iterate through the data loader and convert the features and labels to numpy 
    train_features, train_labels = next(iter(train_dataloader))
    train_dataset_numpy = train_features.numpy()
    train_labels_numpy = train_labels.numpy()

    # Save the train numpy 
    train_dir = os.path.join(destination_folder,f"client_{org_id}", "train")
    os.makedirs(train_dir, exist_ok=True)
    np.save(os.path.join(train_dir, 'train_images.npy'), train_dataset_numpy)
    np.save(os.path.join(train_dir, 'train_labels.npy'), train_labels_numpy)

    print(f"{org_id} train test split success")

# Perform K fold cross-validation split
def cross_val_split(source_folder : str):
    org_id = source_folder.split("/")[-1]
    NUM_FOLDS = 5
    # source_folder = 'data/fl_dataset'
    destination_folder = f'data/fl_dataset/cross_validation_{org_id}'
    # pdb.set_trace
    # MAKE DIRECTORIES
    for split_ind in range(NUM_FOLDS):  
    # make directory for this split
        split_path = os.path.join(destination_folder, 'split' + str(split_ind))
        if not os.path.exists(split_path):
            os.makedirs(split_path)

    # MAKE SPLIT
    copy_dict = {}
    for classes in os.listdir(source_folder):
        if classes[0] == ".":
            continue
        # make directories:
        for split_ind in range(NUM_FOLDS):
            mod_path = os.path.join(destination_folder, 'split' + str(split_ind), classes)
            if not os.path.exists(mod_path):
                os.makedirs(mod_path)

        uni_videos = []
        uni_images = []
        for in_file in os.listdir(os.path.join(source_folder, classes)):
            if in_file[0] == ".":
                continue
            if len(in_file.split(".")) == 3:
                # this is a video
                uni_videos.append(in_file.split(".")[0])
            else:
                # this is an image
                uni_images.append(in_file.split(".")[0])
        # construct dict of file to fold mapping
        inner_dict = {}
        # consider images and videos separately
        for k, uni in enumerate([uni_videos, uni_images]):
            unique_files = np.unique(uni)
            # s is number of files in one split
            s = len(unique_files) // NUM_FOLDS
            for i in range(NUM_FOLDS):
                for f in unique_files[i * s:(i + 1) * s]:
                    inner_dict[f] = i
            # distribute the rest randomly
            for f in unique_files[NUM_FOLDS * s:]:
                inner_dict[f] = np.random.choice(np.arange(5))

        copy_dict[classes] = inner_dict
        for in_file in os.listdir(os.path.join(source_folder, classes)):
            fold_to_put = inner_dict[in_file.split(".")[0]]
            split_path = os.path.join(
                destination_folder, 'split' + str(fold_to_put), classes
            )
            # print(os.path.join(source_folder, classes, file), split_path)
            shutil.copy(os.path.join(source_folder, classes, in_file), split_path)
    
    # pdb.set_trace()
    print(f"cross validation {org_id} success \n")
    return destination_folder

# Split input dataset into N clients
def split_client_dataset(src: str, n : int):
    source_folder = src
    destination_folder = 'data/fl_dataset'

    class_folders = [folder for folder in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, folder))]

    num_classess = len(class_folders)

    # Create the destination folder for each org
    for i in range(n):
        split_folder = os.path.join(destination_folder, f'org_{i+1}')

        os.makedirs(split_folder, exist_ok=True)
        for class_folder in class_folders:
            class_split_folder = os.path.join(split_folder, class_folder)
            os.makedirs(class_split_folder, exist_ok=True)
            

    # Iterate over the files in each folder, distribute evenly between n clients
    for class_folder in class_folders:
        class_path = os.path.join(source_folder, class_folder)
        file_list = os.listdir(class_path)

        
        files_per_split = len(file_list) // n
        remaining_files = len(file_list) % n
        
        start_idx = 0
        end_idx = 0
        split_counts = []

        for i in range(n):
            split_folder = os.path.join(destination_folder, f'org_{i+1}', class_folder)
            split_files = file_list[end_idx:end_idx+files_per_split]

            if remaining_files > 0:
                split_files.append(file_list[end_idx+files_per_split])
                remaining_files -= 1
            
            for file in split_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_folder, file)
                shutil.copy(src, dst)
            
            start_idx = end_idx
            end_idx += files_per_split + (1 if remaining_files > 0 else 0)

    print("client split success\n")

    split_counts = []

    for i in range(n):
        split_folder = os.path.join(destination_folder, f'org_{i+1}')
        class_counts = {}
        
        for class_folder in class_folders:
            class_split_folder = os.path.join(split_folder, class_folder)
            file_count = len(os.listdir(class_split_folder))
            class_counts[class_folder] = file_count
        
        split_counts.append(class_counts)

    for i, counts in enumerate(split_counts):
        print(f"Client {i+1} file counts:")
        for class_folder, count in counts.items():
            print(f"{class_folder}: {count} files")
        print()

    
    for i in range(n):
        split_folder = os.path.join(destination_folder, f'org_{i+1}')

        # Output contains the cross validation folders for each org
        cross_val_split(split_folder)

    for i in range(n):
        source_folder = f"cross_validation_org_{i+1}"
        train_test_split(source_folder,n)



split_client_dataset("data/image_dataset", 3)

