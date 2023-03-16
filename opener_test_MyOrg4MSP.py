import substratools as tools
from torch.utils.data import DataLoader
import pathlib
import numpy as np 

class CovidLUSTestOpener(tools.Opener):
    def get_data(self, folders):
        p = pathlib.Path(folders[0])
        images_data_path = p / list(p.glob("*_images.npy"))[0]
        labels_data_path = p / list(p.glob("*_labels.npy"))[0]

        # load data
        data = {
            "images": np.load(images_data_path),
            "labels": np.load(labels_data_path),
        }

        return data

    def fake_data(self, n_samples=None):
        pass