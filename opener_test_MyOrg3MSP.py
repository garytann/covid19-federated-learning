import substratools as tools
from torch.utils.data import DataLoader


class CovidLUSTestOpener(tools.Opener):
    def get_data(self, folders):
        config = {"center": 1, "train": False}
        return config

    def fake_data(self, n_samples=None):
        pass