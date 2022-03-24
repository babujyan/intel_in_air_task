from utils.general_utils import read_yaml
from torch.utils.data import DataLoader
from data.dataset.fielddataset import FieldDataset


class DataloaderBuilder:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)

    def build(self, mode="train"):
        dataset = FieldDataset(self.config['label_csv_path'],
                               data_dir=self.config['data_dir'],
                               validation_size=self.config['validation_size'],
                               mode=mode)
        return DataLoader(dataset,
                          batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'],
                          num_workers=self.config['num_workers'])
