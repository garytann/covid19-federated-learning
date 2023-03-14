#!/usr/bin/python
import argparse
import pathlib
import pdb
import numpy as np
import os
import torchvision
from typing import OrderedDict

from substra import Client
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec
from substrafl.dependency import Dependency
from substrafl.remote.register import add_metric
from sklearn.metrics import accuracy_score
from substrafl.index_generator import NpIndexGenerator
from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.strategies import FedAvg
from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode
from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment

# Import the custom dataset class
# import covidlus
from covidLUS.dataset import CovidUltrasoundDataset
# from CNN import model, optimizer, criterion, seed
# from torch_dataset import TorchDataset
import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, eval=False):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=not eval)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=not eval)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=not eval)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VGG(nn.Module):

    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

class MyAlgo(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
            use_gpu=True,
        )

class TorchDataset(CovidUltrasoundDataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"]
        self.y = datasamples["labels"]
        self.is_inference = is_inference

    # def __getitem__(self, idx):

    #     if self.is_inference:
    #         x = torch.FloatTensor(self.x[idx][None, ...]) / 255
    #         return x

    #     else:
    #         x = torch.FloatTensor(self.x[idx][None, ...]) / 255

    #         y = torch.tensor(self.y[idx]).type(torch.int64)
    #         y = F.one_hot(y, 10)
    #         y = y.type(torch.float32)

    #         return x, y

    # def __len__(self):
    #     return len(self.x)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# pdb.set_trace()
if not os.path.exists(pathlib.Path.cwd() / "tmp" / "experiment_summaries"):
        os.makedirs(pathlib.Path.cwd() / "tmp" / "experiment_summaries")

N_CLIENTS = 3
seed = 42
torch.manual_seed(seed)
# Number of model updates between each FL strategy aggregation.
NUM_UPDATES = 10
# Number of samples per update.
BATCH_SIZE = 32
# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 3

client_0 = Client(backend_type="subprocess")
client_1 = Client(backend_type="subprocess")
client_2 = Client(backend_type="subprocess")
client_3 = Client(backend_type="subprocess")

clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
    client_3.organization_info().organization_id: client_3,
}

# Store organization IDs
ORGS_ID = list(clients.keys())
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.

# Metric registration for the evaluation of performance of the model on the datasamples
permissions_metric = Permissions(
    public=False, authorized_ids=[ALGO_ORG_ID] + DATA_PROVIDER_ORGS_ID
)

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
metric_deps = Dependency(pypi_dependencies=["numpy==1.23.1", "scikit-learn==1.1.1"])


def accuracy(datasamples, predictions_path):
    config = datasamples
    y_true = datasamples["labels"]
    y_pred = np.load(predictions_path)

    return accuracy_score(y_true, np.argmax(y_pred, axis=1))


metric_key = add_metric(
    client=clients[ALGO_ORG_ID],
    metric_function=accuracy,
    permissions=permissions_metric,
    dependencies=metric_deps,
)

# Directory to the data assets
assets_directory = pathlib.Path.cwd() / "data"
scripts_directory = pathlib.Path.cwd()
empty_path = assets_directory / "empty_datasamples"

# Dict to store the dataset and sample keys
train_dataset_keys = {}
test_dataset_keys = {}

train_datasample_keys = {}
test_datasample_keys = {}

permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])


for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    # pdb.set_trace()
    client = clients[org_id]

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    train_dataset = DatasetSpec(
        name="COVID LUS",
        type="torchDataset",
        data_opener= scripts_directory / f"opener_train_{org_id}.py",
        description= scripts_directory / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    # Add the dataset to the client to provide access to the opener in each organization.
    train_dataset_key = client.add_dataset(train_dataset)
    assert train_dataset_key, "Missing data manager key"

    train_dataset_keys[org_id] = train_dataset_key

     # Add the training data on each organization.
    train_data_sample = DataSampleSpec(
        data_manager_keys=[train_dataset_key],
        test_only=False,
        path=empty_path,
    )
    train_datasample_keys[org_id] = client.add_data_sample(train_data_sample, local=True,)

    # Add the testing data on each organization.
    test_dataset = DatasetSpec(
        name="COVID LUS",
        type="torchDataset",
        data_opener= scripts_directory / f"opener_test_{org_id}.py",
        description= scripts_directory / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    test_dataset_key = client.add_dataset(test_dataset)
    assert test_dataset_key, "Missing data manager key"

    test_dataset_keys[org_id] = test_dataset_key

    test_data_sample = DataSampleSpec(
        data_manager_keys=[test_dataset_key],
        test_only=True,
        path=empty_path,
    )
    test_datasample_keys[org_id] = client.add_data_sample(test_data_sample, local=True,)

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES, 
)

strategy = FedAvg()

aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:
    # pdb.set_trace()
    # Create the Train Data Node (or training task) and save it in a list
    train_data_node = TrainDataNode(
        organization_id=org_id,
        data_manager_key=train_dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    train_data_nodes.append(train_data_node)

test_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Test Data Node (or testing task) and save it in a list
    test_data_node = TestDataNode(
        organization_id=org_id,
        data_manager_key=test_dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_keys=[metric_key],
    )
    test_data_nodes.append(test_data_node)

# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=1)

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
algo_deps = Dependency(pypi_dependencies=["numpy==1.23.1", "torch==1.11.0"])

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    algo=MyAlgo(),
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=algo_deps,
)