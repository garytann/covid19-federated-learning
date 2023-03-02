import argparse
import pathlib
import pdb

from substra import Client
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

# pdb.set_trace()


ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-d",
#     "--data_dir",
#     type=str,
#     # default="./data/splitted/dataset_clientA",
#     help=("Raw data path. Expects 3 python3 cross_val_splitter.py -d ./data/splitted/dataset_clientC -o ./data/cross_validation_C -v ./data/convexor 4 subfolders with classes")
# )
ap.add_argument(
    "-f",
    "--fold",
    type=int,
    default=0,
    help="Number of folds for test dataset"
)
args = vars(ap.parse_args())
NUM_FOLDS = args['fold']

N_CLIENTS = 3

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

# Directory to the data assets
assets_directory = pathlib.Path.cwd() / "data"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    # pdb.set_trace()
    client = clients[org_id]
    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    dataset = DatasetSpec(
        name="COVID LUS",
        type="npy",
        data_opener= "dataset_opener.py",
        description= "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing dataset key"

     # Add the training data on each organization.
    train_data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=False,
        path=assets_directory / f"cross_validation_{i+1}",
    )
    train_datasample_keys[org_id] = client.add_data_sample(train_data_sample)

    # Add the testing data on each organization.
    test_data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        test_only=True,
        path=assets_directory / f"cross_validation_{i+1}" / f"split{NUM_FOLDS}",
    )
    test_datasample_keys[org_id] = client.add_data_sample(test_data_sample)

    pdb.set_trace()

