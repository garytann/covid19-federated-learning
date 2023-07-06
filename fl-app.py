import argparse
import os
import pdb
import collections

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from imutils import paths
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from model import get_model
# Suppress logging
tf.get_logger().setLevel('ERROR')

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    '-d', '--data_dir', required=True, help='path to input dataset'
)
ap.add_argument('-m', '--model_dir', type=str, default='models/')
ap.add_argument(
    '-f', '--fold', type=int, default='0', help='fold to take as test data'
)
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
ap.add_argument('-ep', '--epochs', type=int, default=2)
ap.add_argument('-bs', '--batch_size', type=int, default=16)
ap.add_argument('-t', '--trainable_base_layers', type=int, default=1)
ap.add_argument('-iw', '--img_width', type=int, default=224)
ap.add_argument('-ih', '--img_height', type=int, default=224)
ap.add_argument('-id', '--model_id', type=str, default='vgg_base')
ap.add_argument('-ls', '--log_softmax', type=bool, default=False)
ap.add_argument('-n', '--model_name', type=str, default='test')
ap.add_argument('-hs', '--hidden_size', type=int, default=64)
args = vars(ap.parse_args())

# Initialize hyperparameters
DATA_DIR = args['data_dir']
MODEL_NAME = args['model_name']
FOLD = args['fold']
MODEL_DIR = os.path.join(args['model_dir'], MODEL_NAME, f'fold_{FOLD}')
LR = args['learning_rate']
EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
MODEL_ID = args['model_id']
TRAINABLE_BASE_LAYERS = args['trainable_base_layers']
IMG_WIDTH, IMG_HEIGHT = args['img_width'], args['img_height']
LOG_SOFTMAX = args['log_softmax']
HIDDEN_SIZE = args['hidden_size']

# CLIENT_ID = "CLIENT_1"
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
CLIENT_ID = ["client"]  # Replace with appropriate client IDs


def make_federated_data(client_data, client_ids):
  return [
      f_preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

def f_preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    print(f'what is element: {element}')
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 224, 224, 3]),
        y=tf.reshape(element['label'], [-1, 3]))

  return dataset.repeat(EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def input_spec():
    return (
        tf.TensorSpec([None, 224,224,3], tf.float64),
        tf.TensorSpec([None, 3], tf.int64)
    )

def model_fn():
    model = get_model()
    return tff.learning.models.from_keras_model(
        model,
        input_spec=input_spec(),
        loss=tf.keras.losses.CategoricalCrossentropy() if not LOG_SOFTMAX else (
        lambda labels, targets: tf.reduce_mean(
            tf.reduce_sum(
                -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
                axis=1
            )
        )
    ),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

def preprocess(imagePaths, fold):
    print(f'selected fold: {fold}')
    train_labels, test_labels = [], []
    train_data, test_data = [], []

    for imagePath in imagePaths:
        path_parts = imagePath.split(os.path.sep)
        # extract the split
        train_test = path_parts[-3][-1]
        # extract the class label from the filename
        label = path_parts[-2]
        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

        # update the data and labels lists, respectively
        if train_test == str(fold):
            test_labels.append(label)
            test_data.append(image)
            # test_files.append(path_parts[-1])
        else:
            train_labels.append(label)
            train_data.append(image)
    print(f'processing complete...')
    return train_labels, train_data, test_labels, test_data

def convert_numpy(train_data, train_labels, test_data, test_labels):
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
    train_data = np.array(train_data) / 255.0
    test_data = np.array(test_data) / 255.0
    train_labels_text = np.array(train_labels)
    test_labels_text = np.array(test_labels)
    return train_data, train_labels_text, test_data, test_labels_text


imagePaths = list(paths.list_images(DATA_DIR))
# model = get_model()


train_labels, train_data, test_labels, test_data = preprocess(imagePaths=imagePaths, fold = FOLD)

print(
    f'\nNumber of training samples: {len(train_labels)} \n'
    f'Number of testing samples: {len(test_labels)}'
)


train_data, train_labels_text, test_data, test_labels_text = convert_numpy(train_data, train_labels, test_data, test_labels)

num_classes = len(set(train_labels))


# perform one-hot encoding on the labels
lb = LabelBinarizer()
lb.fit(train_labels_text)

train_labels = lb.transform(train_labels_text)
test_labels = lb.transform(test_labels_text)

trainX = train_data
trainY = train_labels
testX = test_data
testY = test_labels

# convert data and labels numpy array into tensor
# Convert your NumPy arrays to TensorFlow tensors
# trainX = tf.constant(trainX, dtype=tf.float32)
# trainY = tf.constant(trainY, dtype=tf.float32)
# testX = tf.constant(testX, dtype=tf.float32)
# testY = tf.constant(testY, dtype=tf.float32)

total_image_count = len(trainX)
image_per_set = int(np.floor(total_image_count/2))
# Create TensorFlow datasets
# train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
# test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))


# train_dataset = train_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

client_train_dataset = collections.OrderedDict()

data = collections.OrderedDict((('label', trainY), ('pixels', trainX)))
client_train_dataset[CLIENT_ID[0]] = data

train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

example_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
sample_client = [train_dataset.client_ids[0]]
federated_data = make_federated_data(train_dataset, sample_client)
# example_element = next(iter(example_dataset))

# preprocessed_example_dataset = f_preprocess(example_dataset)


# sampling the example dataset
# sample_batch = tf.nest.map_structure(lambda x: x.numpy(),next(iter(preprocessed_example_dataset)))

# train_dataset = tff.simulation.datasets.TestClientData.from_clients_and_tf_fn(CLIENT_ID, lambda x: train_dataset)
# pdb.set_trace()



trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0)
)


state = trainer.initialize()
train_hist = []
for i in range(EPOCHS):
    state, metrics = trainer.next(state, federated_data)
    train_hist.append(metrics)

    print(f"\rRun {i+1}/{EPOCHS}", end="")

pdb.set_trace()


# evaluator = tff.learning.build_federated_evaluation(model_fn)
# federated_metrics = evaluator(state.model, test_dataset)
# print(f"{federated_metrics}")





