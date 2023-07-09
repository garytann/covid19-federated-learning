import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam

from utils import *
from model import *
import pdb

# Defining the base parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
DATA_DIR = "../data/cross_validation"
FOLD = 0
LR = 0.01
LOG_SOFTMAX = False
HIDDEN_SIZE = 64
NUM_ROUND = 10

# Loading and transforming the dataset
imagePaths = list(paths.list_images(DATA_DIR))

train_data, train_labels_text, test_data, test_labels_text = load(imagePaths=imagePaths,
                                                    fold = FOLD,
                                                    img_width=IMG_WIDTH,
                                                    img_height=IMG_HEIGHT
                                                    )

# perform one-hot encoding on the labels
lb = LabelBinarizer()
lb.fit(train_labels_text)

train_labels = lb.transform(train_labels_text)
test_labels = lb.transform(test_labels_text)

trainX = train_data
trainY = train_labels
testX = test_data
testY = test_labels

#create clients
clients = create_clients(trainX, trainY, num_clients=1, initial='client')

#process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

#process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices(
    (testX, testY)).batch(len(testY))

# Initialize the hyperparameter for local training

# number of global aggregation
comms_round = 10

# init the global model 
global_model = CovidLUSModel.get_model()

# metrics = Metrics((testX, testY), global_model)
metrics = ['accuracy']

optimizer = Adam(learning_rate=LR, decay = LR / comms_round)

loss = (
    tf.keras.losses.CategoricalCrossentropy() if not LOG_SOFTMAX else (
        lambda labels, targets: tf.reduce_mean(
            tf.reduce_sum(
                -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
                axis=1
            )
        )
    )
)

#commence global training loop
for comm_round in range(comms_round):
    print(f'initiating communication round {comm_round}')
            
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)
    
    #loop through each client and create new local model
    for client in client_names:
        local_model = CovidLUSModel.get_model()
        local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        '''
        before fitting the model into the local data, peform
        image augmentation first by looping through the batched dataset
        '''
        train_x = []
        train_y = []
        for batch_x, batch_y in clients_batched[client]:
            train_x.append(batch_x)
            train_y.append(batch_y)
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        #fit local model with client's data
        # local_model.fit(clients_batched[client],epochs=10, verbose=1)
        local_model.fit(trainAug.flow(train_x, train_y, batch_size = 8),
                        epochs=2, 
                        verbose=1,
                        callbacks=[earlyStopping, reduce_lr_loss])
        
        #scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        # pdb.set_trace()
        #clear session to free memory after each communication round
        clear_session()
        
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, 
                                             global_model, 
                                             comm_round)












