#!/usr/bin/python
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pathlib 
from imutils import paths
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pdb
import PIL
import torchvision.transforms.functional as transform


data_path = pathlib.Path.cwd() / "data"
train_image_data_path = pathlib.Path(data_path) / "train_dataset"
test_image_data_path = pathlib.Path(data_path) / "test_dataset"
train_label_path = pathlib.Path(data_path) / "train_annotations.csv"
test_label_path = pathlib.Path(data_path) / "test_annotations.csv"


class CovidUltrasoundDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
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
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

train_covid_dataset = CovidUltrasoundDataset(annotations_file=train_label_path,
                                      img_dir=train_image_data_path
                                      )

test_covid_dataset = CovidUltrasoundDataset(annotations_file=test_label_path,
                                      img_dir=test_image_data_path
                                      )
    
train_dataloader = DataLoader(train_covid_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_covid_dataset, batch_size=64, shuffle=True)

# # Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0]

# Rotating the channels of the image
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)
label = train_labels[0]
# pdb.set_trace()
plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
plt.savefig('test.png')
print(f"Label: {label}")