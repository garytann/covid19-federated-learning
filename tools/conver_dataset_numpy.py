import os 
import numpy as np 
import pathlib
import pdb
import cv2
from imutils import paths
from PIL import Image


DATASET_DIR = "data/cross_validation_1"

train_labels, test_labels = [], []
train_data, test_data = [], []
IMG_WIDTH = 224
IMG_HEIGHT = 224

# split0 will be used for testing dataset
FOLD = 0 
# Path to the dataset
data_path = pathlib.Path.cwd() / "data"

# Actual path to the image dataset
image_data_path = pathlib.Path(data_path) / "cross_validation_1"

imagePaths = list(paths.list_images(image_data_path))

N_CLIENTS = 3


for imagepath in imagePaths:
    path_parts = path_parts = imagepath.split(os.path.sep)
    train_test = path_parts[-3][-1]
    label = path_parts[-2]
    # image = Image.open(imagepath)
    # # Resize the image to 224x224 pixels
    # new_size = (224, 224)
    # image = image.resize(new_size)
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    if train_test == str(FOLD):
        test_labels.append(label)
        test_data.append(image)
    else:
        train_labels.append(label)
        train_data.append(image)

# Prepare data for model
print(
    f'\nNumber of training samples: {len(train_labels)} \n'
    f'Number of testing samples: {len(test_labels)}'
)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
train_data = np.array(train_data, dtype='uint8') / 255.0
train_data = np.expand_dims(train_data, axis = 0)
# train_data = train_data.transpose(0, 3, 1, 2)

test_data = np.array(test_data,dtype='uint8') / 255.0
test_data = np.expand_dims(test_data, axis = 0)
# test_data = test_data.transpose(0, 3, 1, 2)

train_labels_text = np.array(train_labels,dtype='uint8')
test_labels_text = np.array(test_labels, dtype='uint8')
# pdb.set_trace()

# Creating the train data and label for org X
os.makedirs(str(data_path / f"org_{1}/train"), exist_ok=True)
filename = data_path / f"org_{1}/train/train_images.npy"
np.save(str(filename), train_data)
filename = data_path / f"org_{1}/train/train_labels.npy"
np.save(str(filename), train_labels_text)

# Creating the test data and label for org X
os.makedirs(str(data_path / f"org_{1}/test"), exist_ok=True)
filename = data_path / f"org_{1}/test/test_images.npy"
np.save(str(filename), test_data)
filename = data_path / f"org_{1}/test/test_labels.npy"
np.save(str(filename), test_labels_text)


# image_ls = []
# for data in os.listdir(DATASET_DIR):v
#     for classes in os.listdir(os.path.join(DATASET_DIR, data)):
#         for images in os.listdir(os.path.join(DATASET_DIR, data, classes)):
#             image_ls.append(images)

# print(len(image_ls))