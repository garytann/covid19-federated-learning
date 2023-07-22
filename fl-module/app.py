import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import uvicorn
from fastapi import FastAPI, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import aiohttp
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from utils import *
from model import *
import pdb


# Defining the base parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
DATA_DIR = "../data/fl_assets/cross_validation_org_1"
FOLD = 4
LR = 0.01
LOG_SOFTMAX = False
HIDDEN_SIZE = 64
NUM_ROUND = 10
CLIENT = "client_1"

client_weights_filepath_list = []   # Contains Array of Client Weights filepath
global_dataset_size = 0             # Contains Sum of client dataset size
client_dataset_size_list = []       # Contains Array of client dataset size
# Loading and transforming the dataset
imagePaths = list(paths.list_images(DATA_DIR))


# Loading the dataset
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

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "200"}

@app.get("/client/train")
async def client_train():

    # #create clients
    clients = create_clients(trainX, trainY, num_clients=1, initial=f'{CLIENT}')

    #process and batch the training data for each client
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = batch_data(data)

    # initialize hyperparameters for local training
    metrics = ['accuracy']
    epoch = 10

    initial_learning_rate = 1e-4
    # final_learning_rate = 0.0001
    # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epoch)
    # steps_per_epoch = int(len(trainX)/8)
    
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #             initial_learning_rate=initial_learning_rate,
    #             decay_steps=steps_per_epoch,
    #             decay_rate=learning_rate_decay_factor,
    #             staircase=True
    #             )
    optimizer = Adam(learning_rate=initial_learning_rate, decay = initial_learning_rate / epoch)

    loss = tf.keras.losses.CategoricalCrossentropy()
    # loss = (
    #     tf.keras.losses.CategoricalCrossentropy() if not LOG_SOFTMAX else (
    #         lambda labels, targets: tf.reduce_mean(
    #             tf.reduce_sum(
    #                 -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
    #                 axis=1
    #             )
    #         )
    #     )
    # )
    print(f'client: initiate client model training')
    # initiate the local model
    if os.path.exists('global_model.h5'):
        print(f'retrieving global model...')
        local_model = tf.keras.models.load_model('global_model.h5')
    else:
        local_model = CovidLUSModel.get_model()

    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)

    for client in client_names:
        # inititalize model
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
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

        # train local model with client's data
        local_model.fit(trainAug.flow(train_x, train_y, batch_size = 8),
                        epochs=epoch, 
                        verbose=1,
                        # validation_data = 0.1,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

        # save the client model weights
        local_model.save_weights(f'{CLIENT}_weights_{FOLD}.h5', save_format='h5')

    clear_session()
    return {"client: completed local model training"}


@app.post("/client/send_weights")
async def client_send_weights():
    model_file = open(f'{CLIENT}_weights_{FOLD}.h5', 'rb')
    datasize = len(trainX)
    metadata = {
        'datasize': datasize
    }
    # res = requests.post('http://localhost:8000/server/receive_client_weights', data = model_file, params = metadata)
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8000/server/receive_client_weights', data = model_file, params = metadata) as res:
            print(f'client: response from server:{res.text}')

    return {"client: weights sent to server"}


@app.post("/client/receive_global_weights")
async def client_receive_global_weights(request : Request):
    file = await request.body()
    print(f'client: received global model')
    filename = "global_model.h5"
    with open(filename, 'wb') as f:
        f.write(file)
    f.close()
    return {"client: updated weights received"}

@app.post("/server/receive_client_weights")
async def server_receive_client_weights(request : Request):
    file = await request.body()
    metadata = dict(request.query_params)

    global global_dataset_size
    global client_dataset_size_list

    client_dataset_size = metadata['datasize']
    print(f"client dataset size: {client_dataset_size}")

    filename = f"{CLIENT}_weights_{FOLD}.h5"
    with open(filename, 'wb') as f:
        f.write(file)
    f.close()

    client_weights_filepath_list.append(filename)
    client_dataset_size_list.append(int(client_dataset_size))
    global_dataset_size = global_dataset_size + int(client_dataset_size)

    await server_aggregation()
    
    return {"server: clients weights received"}


async def server_aggregation():
    print(f'server: starting global model aggregation')
    # # process and batch the test set
    test_batched = tf.data.Dataset.from_tensor_slices((testX, testY)).batch(len(testY))
   
    # comm_round = 1
    # for _ in range(comm_round):
    if os.path.exists('global_model.h5'):
        global_model = tf.keras.models.load_model("global_model.h5")
    else:
        global_model = CovidLUSModel.get_model()

    print("Aggregating client weights")
    
    # determine client dataset contribution scale
    weight_scaling_factor_list = weight_scaling_factor() 
    
    # Scale client model weight by client dataset contribution scale
    scaled_client_weights = scale_model_weight(weight_scaling_factor_list, global_model) 
    
    # Sum scaled 
    sum_scaled_weights(scaled_client_weights, global_model) 

    # test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, 
                                            global_model, 
                                            # comm_round
                                            )
        print("server: model aggregated")
        # await server_send_updated_weights()
    
    print('server: updated weights sent to client')

async def server_send_updated_weights():
    print(f'sending updated global weights to client')
    file = open("updated_global_weights.h5", 'rb')
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8000/client/receive_global_weights', data = file) as res:
            print(f'server: response from client:{res.text}')


@app.post("/client/inference")
async def inference(file: UploadFile):
    print(f'client: starting inference')
    contents = await file.read()
    model = tf.keras.models.load_model("global_model.h5", compile=False)

    # load the image
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    resized_image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    testX = np.array(resized_image) / 255.0
    testX = np.reshape(testX, [1, 224,224,3])/255

    # make predictions on the testing set
    logits = model.predict(testX)
    print(f'logits:{logits}')
    predicted_class = np.argmax(logits, axis=1)
    response_data = {'logits:', str(predicted_class)}
    json_res = jsonable_encoder(response_data)
    
    return json_res

def weight_scaling_factor():
    global global_dataset_size
    # print(client_dataset_size_list)
    
    weight_scaling_factor_list = [client_datasize / global_dataset_size for client_datasize in client_dataset_size_list]
    print(weight_scaling_factor_list)
    return weight_scaling_factor_list

def scale_model_weight(weight_scaling_factor_list, model):
    scaled_client_weights = []
    # print(client_weights_filepath_list)
    for i, scaling_factor in enumerate(weight_scaling_factor_list):
        print(i)
        filepath = client_weights_filepath_list[int(i)]
        model.load_weights(filepath)
        client_weight = model.get_weights()
        client_weight = np.array(client_weight, dtype='object')
        scaled_client_weights.append(client_weight * scaling_factor)
    return scaled_client_weights

def sum_scaled_weights(scaled_client_weights, model):
    global_weight = scaled_client_weights[0]
    for client_weight in scaled_client_weights[1:]:
        global_weight = np.add(global_weight, client_weight)
    model.set_weights(global_weight)
    # model.save_weights("updated_global_weights.h5")
    model.save("global_model.h5")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000)


'''
Deprecated code
'''
# #create clients
# clients = create_clients(trainX, trainY, num_clients=1, initial='client')

# #process and batch the training data for each client
# clients_batched = dict()
# for (client_name, data) in clients.items():
#     clients_batched[client_name] = batch_data(data)

# # process and batch the test set
# test_batched = tf.data.Dataset.from_tensor_slices(
#     (testX, testY)).batch(len(testY))

# # Initialize the hyperparameter for local training

# # number of global aggregation
# comms_round = 2

# # init the global model 
# global_model = CovidLUSModel.get_model()

# metrics = Metrics((testX, testY), global_model)
# metrics = ['accuracy']

# optimizer = Adam(learning_rate=LR, decay = LR / comms_round)

# loss = (
#     tf.keras.losses.CategoricalCrossentropy() if not LOG_SOFTMAX else (
#         lambda labels, targets: tf.reduce_mean(
#             tf.reduce_sum(
#                 -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
#                 axis=1
#             )
#         )
#     )
# )

# #commence global training loop
# for comm_round in range(comms_round):
#     print(f'initiating communication round {comm_round}')

#     # get the global model's weights - will serve as the initial weights for all local models
#     # global_weights = global_model.get_weights()
    
#     #initial list to collect local model weights after scalling
#     scaled_local_weight_list = list()

#     #randomize client data - using keys
#     client_names= list(clients_batched.keys())
#     random.shuffle(client_names)
    
#     #loop through each client and create new local model
#     for client in client_names:
#         local_model = CovidLUSModel.get_model()
#         local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
#         #set local model weight to the weight of the global model
#         local_model.set_weights(global_weights)

#         '''
#         before fitting the model into the local data, peform
#         image augmentation first by looping through the batched dataset
#         '''
#         train_x = []
#         train_y = []
#         for batch_x, batch_y in clients_batched[client]:
#             train_x.append(batch_x)
#             train_y.append(batch_y)
#         train_x = np.concatenate(train_x)
#         train_y = np.concatenate(train_y)
#         #fit local model with client's data
#         # local_model.fit(clients_batched[client],epochs=10, verbose=1)
#         local_model.fit(trainAug.flow(train_x, train_y, batch_size = 8),
#                         epochs=2, 
#                         verbose=1,
#                         callbacks=[earlyStopping, reduce_lr_loss, mcp_save])
        
#         #scale the model weights and add to list
#         scaling_factor = weight_scalling_factor(clients_batched, client)
#         scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
#         scaled_local_weight_list.append(scaled_weights)
#         local_model.save_weights('client_weights',save_format="h5")
        
#         # pdb.set_trace()
#         #clear session to free memory after each communication round
#         clear_session()

#     # saving the local model weights

#     #to get the average over all the local model, we simply take the sum of the scaled weights
#     average_weights = sum_scaled_weights(scaled_local_weight_list)
    
#     #update global model 
#     global_model.set_weights(average_weights)

#     #test global model and print out metrics after each communications round
#     for(X_test, Y_test) in test_batched:
#         global_acc, global_loss = test_model(X_test, Y_test, 
#                                              global_model, 
#                                              comm_round)

# # saving the global model weights at the end of the process
#     global_model.save_weights('server_weights',save_format="h5")







