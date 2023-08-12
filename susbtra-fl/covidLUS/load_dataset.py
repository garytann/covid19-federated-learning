#!/usr/bin/python
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pathlib 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np 
from PIL import Image
import pdb
from torchvision.io import read_image, ImageReadMode


class CovidUltrasoundDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 transform=None, 
                 target_transform=None,
                 ):
        
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # for data loader 
    def __len__(self):
        return len(self.img_labels)
    
    # for data loader
    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, str(label) ,self.img_labels.iloc[idx, 0])
        image = Image.open(fp=img_path)
        image_size = (224,224)
        image = image.resize(image_size)
        transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                    # transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, interpolation=False, fill=0),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    # transforms.Lambda(lambda x: x[0]) # remove color channel
                ])
        img = transform(image)

        # if self.transform:
        #     img = self.transform(img)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return img, label

# if __name__ == "__main__":
# for i in range(3):
#     data_path = pathlib.Path.cwd() / "data"
#     train_image_data_path = pathlib.Path(data_path) / f"train_dataset_{i+1}"
#     test_image_data_path = pathlib.Path(data_path) / f"test_dataset_{i+1}"
#     train_label_path = pathlib.Path(data_path) / f"train_annotation_{i+1}.csv"
#     test_label_path = pathlib.Path(data_path) / f"test_annotation_{i+1}.csv"
#     # Load the train dataset 
#     train_covid_dataset = CovidUltrasoundDataset(
#                                     annotations_file=train_label_path,
#                                     img_dir=train_image_data_path,
#                                     )
#     print(f'len of train set of org_{i+1}: {len(train_covid_dataset)}')

#     train_dataloader = DataLoader(
#                             train_covid_dataset, 
#                             batch_size=len(train_covid_dataset), 
#                             shuffle=True
#                             )

#     # Iterate through the data loader and convert the features and labels to numpy 
#     train_features, train_labels = next(iter(train_dataloader))
#     train_dataset_numpy = train_features.numpy()
#     train_labels_numpy = (train_labels-1).numpy()
#     # (660, 224, 224) shape of the image
#     train_labels_numpy = train_labels_numpy.astype(np.uint8)
#     # import pdb
#     # pdb.set_trace()
#     # img = train_features[0]

#     # # Rotating the channels of the image
#     # img = img.swapaxes(0,1)
#     # img = img.swapaxes(1,2)
#     # # label = train_labels[0]

#     # # Showing a sample of the train features 
#     # plt.imshow(img.squeeze())
#     # # plt.show()
#     # plt.savefig('img_sample.png')
#     # # print(f"Label: {label}")

#     # pdb.set_trace()
#     # Save the numpy to a directory 
#     os.makedirs(str(data_path / f"org_{i+1}/train"), exist_ok=True)
#     filename = data_path / f"org_{i+1}/train/train_images.npy"
#     np.save(str(filename), train_dataset_numpy)
#     filename = data_path / f"org_{i+1}/train/train_labels.npy"
#     np.save(str(filename), train_labels_numpy)
    
#     # Load the test dataset 
#     test_covid_dataset = CovidUltrasoundDataset(
#                                         annotations_file=test_label_path,
#                                         img_dir=test_image_data_path
#                                         )
#     print(f'len of test set of org_{i+1}: {len(test_covid_dataset)}')
   
#     test_dataloader = DataLoader(
#                         test_covid_dataset, 
#                         batch_size=len(test_covid_dataset), 
#                         shuffle=True
#                     )
    
#     # Iterate through the data loader and convert the features and labels to numpy 
#     test_features, test_labels = next(iter(test_dataloader))
#     test_dataset_numpy = test_features.numpy()
#     test_labels_numpy = test_labels.numpy()

#     # pdb.set_trace()

#     # Save the numpy to directory 
#     os.makedirs(str(data_path / f"org_{i+1}/test"), exist_ok=True)
#     filename = data_path / f"org_{i+1}/test/test_images.npy"
#     np.save(str(filename), test_dataset_numpy)
#     filename = data_path / f"org_{i+1}/test/test_labels.npy"
#     np.save(str(filename), test_labels_numpy)

# print('dataset loaded successfully')

    # # # Display image and label
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0]

    # # Rotating the channels of the image
    # img = img.swapaxes(0,1)
    # img = img.swapaxes(1,2)
    # label = train_labels[0]
    # # pdb.set_trace()
    # plt.imshow(img.squeeze())
    # # plt.show()
    # plt.savefig('test.png')
    # print(f"Label: {label}")