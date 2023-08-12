#!/usr/bin/python
import argparse
import pathlib
import pdb
import numpy as np
import os
import torchvision
from typing import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File
from tools import dataset_preparation
import shutil
import uvicorn

# Substra Imports
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
from substrafl.strategies import FedAvg, SingleOrganization
from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode
from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment

# Import the custom dataset class
# import covidlus
# from covidLUS.load_dataset import CovidUltrasoundDataset
# from CNN import model, optimizer, criterion, seed
# from torch_dataset import TorchDataset
import torch
from torch import nn
import torch.nn.functional as F

# Torch dataset required 
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["images"]
        self.y = datasamples["labels"]
        # pdb.set_trace()
        # self.y = torch.tensor(self.y)
        # self.y_onehot = torch.zeros(self.y.shape[0],3)
        self.is_inference = is_inference

    def __getitem__(self, idx):

        if self.is_inference:
            x = (torch.FloatTensor(self.x[idx][None, ...]) / 255)
            return x

        else:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255
            # y = torch.tensor(self.y[idx]).random_(0,2)
            y = torch.tensor(self.y[idx])
            # y = torch.tensor(self.y[idx]).type(torch.int64)
            # pdb.set_trace()
            # x.to(device)
            # y = F.one_hot(y, 10)
            y = F.one_hot(y.to(torch.int64), 3)
            y = y.type(torch.float32)
            # y.to(device)
            return x, y

    def __len__(self):
        return len(self.x)

# VGG16 model
class VGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG16, self).__init__()
        self.features = models.vgg16(pretrained=True).features  # Load pre-trained VGG16 features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Adaptive average pooling layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.squeeze(1)
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# define accuracy function for datasamples
def accuracy(datasamples, predictions_path):
    # config = datasamples
    y_true = datasamples["labels"]
    y_pred = np.load(predictions_path)

    return accuracy_score(y_true, np.argmax(y_pred, axis=1))


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "200"}

@app.post("/upload-folder")
async def upload_folder(folder: UploadFile = File(...)):
    # Create a temporary directory to store the uploaded folder
    temp_dir = "/path/to/temp/directory"
    temp_folder = os.path.join(temp_dir, folder.filename)
    os.makedirs(temp_folder, exist_ok=True)

    try:
        # Save the uploaded folder to the temporary directory
        file_content = await folder.read()
        temp_zip_file = os.path.join(temp_dir, folder.filename + ".zip")
        with open(temp_zip_file, "wb") as temp_zip:
            temp_zip.write(file_content)

        # Extract the contents of the uploaded folder
        shutil.unpack_archive(temp_zip_file, temp_folder)

        # Process the contents of the folder
        # Example: Print the file paths in the folder
        for root, dirs, files in os.walk(temp_folder):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)

        # Perform further processing on the extracted folder contents

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_folder)
        os.remove(temp_zip_file)

    return {"message": "Folder uploaded and processed successfully."}

@app.post("/process-assets")
async def process_data(path : str, num_clients : int = 3):
    # file_content = await file.read()
    if (dataset_preparation.split_client_dataset(path, num_clients)):
        return {"status": "success"}
    else:
        return {"status": "success"}

@app.post("/train")
def federated_training(num_clients : int, num_updates : int, num_rounds : int):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument('--num_client', type=int, default=3)
    # ap.add_argument('--num_updates', type=int, default=10)
    # ap.add_argument('--num_rounds', type=int, default=3)
    # args = vars(ap.parse_args())

    NUM_CLIENTS = num_clients
    NUM_UPDATES = num_updates
    NUM_ROUNDS = num_rounds

    seed = 42
    torch.manual_seed(seed)
    # Number of model updates between each FL strategy aggregation.
    NUM_UPDATES = NUM_UPDATES
    # Number of samples per update.
    BATCH_SIZE = 16
    # A round is defined by a local training step followed by an aggregation operation
    NUM_ROUNDS = NUM_ROUNDS

    model = VGG16()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(pathlib.Path.cwd() / "tmp" / "experiment_summaries"):
            os.makedirs(pathlib.Path.cwd() / "tmp" / "experiment_summaries")
    clients = {}
    for _ in range(NUM_CLIENTS + 1):
        client = Client(backend_type="subprocess") 
        # client = Client(backend_type="docker") 
        # client.login(username="substra", password="substra")
        clients[client.organization_info().organization_id] = client
        ORGS_ID = list(clients.keys())
        # script_path = pathlib.Path.cwd() / f"opener_test_{ORGS_ID[-1]}.py"
        # if not os.path.exists(script_path):
        script = f'''
import substratools as tools
import pathlib
import numpy as np
class CovidLUSTrainOpener(tools.Opener):
def get_data(self, folders):
    p = pathlib.Path(folders[0])
    images_data_path = p / list(p.glob("*_images.npy"))[0]
    labels_data_path = p / list(p.glob("*_labels.npy"))[0]

    # load data
    data = {{
        "images": np.load(images_data_path),
        "labels": np.load(labels_data_path),
    }}

    return data

def fake_data(self, n_samples=None):
    pass

    '''
        # Save the generated script as a Python file
        script_test = f'opener_test_{ORGS_ID[-1]}.py'
        script_train = f'opener_train_{ORGS_ID[-1]}.py'
        with open(script_test, 'w') as f:
            f.write(script)
        with open(script_train, 'w') as f:
            f.write(script)

    # Store organization IDs
    ORGS_ID = list(clients.keys())
    ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
    DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.
    app_directory = pathlib.Path.cwd() / "app.py"

    # Metric registration for the evaluation of performance of the model on the datasamples
    permissions_metric = Permissions(
        public=False, authorized_ids=[ALGO_ORG_ID] + DATA_PROVIDER_ORGS_ID
    )

    # The Dependency object is instantiated in order to install the right libraries in
    # the Python environment of each organization.
    metric_deps = Dependency(pypi_dependencies=["numpy==1.23.1", 
                                                "scikit-learn==1.1.1", 
                                                "uvicorn==0.22.0", 
                                                "fastapi==0.98.0",
                                                "python-multipart==0.0.5"
                                                ],
                            local_code=[app_directory]
                            )
    
    pdb.set_trace()

    metric_key = add_metric(
        client=clients[ALGO_ORG_ID],
        metric_function=accuracy,
        permissions=permissions_metric,
        dependencies=metric_deps,
    )

    # Directory to the data assets
    assets_directory = pathlib.Path.cwd() / "data" / "fl_assets"
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
            type="npy",
            data_opener= scripts_directory / f"opener_train_{org_id}.py",
            description= scripts_directory / "description.md",
            permissions=permissions_dataset,
            logs_permission=permissions_dataset,
        )
        # pdb.set_trace()
        # Add the dataset to the client to provide access to the opener in each organization.
        train_dataset_key = client.add_dataset(train_dataset)
        assert train_dataset_key, "Missing data manager key"

        train_dataset_keys[org_id] = train_dataset_key

        # Add the training data on each organization.
        train_data_sample = DataSampleSpec(
            data_manager_keys=[train_dataset_key],
            test_only=False,
            path= assets_directory / f"client_org_{i+1}" / "train",
        )
        train_datasample_keys[org_id] = client.add_data_sample(train_data_sample, local=True,)

        # Add the testing data on each organization.
        test_dataset = DatasetSpec(
            name="COVID LUS",
            type="npy",
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
            path=assets_directory / f"client_org_{i+1}" / "test",
        )
        test_datasample_keys[org_id] = client.add_data_sample(test_data_sample, local=True,)

    index_generator = NpIndexGenerator(
        batch_size=BATCH_SIZE,
        num_updates=NUM_UPDATES, 
    )

    strategy = FedAvg()
    # strategy = SingleOrganization()

    aggregation_node = AggregationNode(ALGO_ORG_ID)

    train_data_nodes = list()

    class MyAlgo(TorchFedAvgAlgo):
        def __init__(self):
            super().__init__(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                index_generator=index_generator,
                dataset=TorchDataset,
                seed=seed,
                use_gpu=False,
            )

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
    algo_deps = Dependency(pypi_dependencies=["numpy==1.23.1", 
                                                "scikit-learn==1.1.1", 
                                                "uvicorn==0.22.0", 
                                                "fastapi==0.98.0",
                                                "python-multipart==0.0.5"
                                            ],
                            local_code=[app_directory]
                            )

    def plot_results():
        performances_df = pd.DataFrame(client.get_performances(compute_plan.key).dict())
        print("\nPerformance Table: \n")
        print(performances_df[["worker", "round_idx", "identifier" "performance"]])
        plt.style.use('ggplot')
        plt.figure()
        plt.title("Test dataset results")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")

        for id in DATA_PROVIDER_ORGS_ID:
            df = performances_df.query(f"worker == '{id}'")
            plt.plot(df["round_idx"], df["performance"], label=id)

        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('federated_result.png')

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

    plot_results()

    # print('federated training completed..')
    return {"status": "200 federated training completed"}
# federated_training()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
        





    