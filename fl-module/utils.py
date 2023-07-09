import cv2
import numpy as np 
import os
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pdb

def load(imagePaths, fold, img_width, img_height):
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
        image = cv2.resize(image, (img_width, img_height))

        # update the data and labels lists, respectively
        if train_test == str(fold):
            test_labels.append(label)
            test_data.append(image)
            # test_files.append(path_parts[-1])
        else:
            train_labels.append(label)
            train_data.append(image)
    print(f'load complete...')

    '''
    Converting the list to numpy array with normalizing the pixel values
    '''
    train_data = np.array(train_data) / 255.0
    test_data = np.array(test_data) / 255.0
    train_labels_text = np.array(train_labels)
    test_labels_text = np.array(test_labels)

    return train_data, train_labels_text, test_data, test_labels_text


def fix_layers(model, num_flex_layers: int = 1):
    """
    Receives a model and freezes all layers but the last num_flex_layers ones.

    Arguments:
        model {tensorflow.python.keras.engine.training.Model} -- model

    Keyword Arguments:
        num_flex_layers {int} -- [Number of trainable layers] (default: {1})

    Returns:
        Model -- updated model
    """
    num_layers = len(model.layers)
    for ind, layer in enumerate(model.layers):
        if ind < num_layers - num_flex_layers:
            layer.trainable = False

    return model

def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    # image_list = trainAug.flow(image_list, label_list)
    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}

def batch_data(data_shard, bs=8):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))

    return dataset.shuffle(len(label)).batch(bs)

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss
