import substratools as tools
from torch.utils.data import DataLoader

# from flamby.datasets.fed_tcga_brca import FedTcgaBrca


class CovidLUSTrainOpener(tools.Opener):
    def get_data(self, folders):
        config = {"center": 0, "train": True}
        return config

    def fake_data(self, n_samples=None):
        pass