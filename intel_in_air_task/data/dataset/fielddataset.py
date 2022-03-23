from lib2to3.pgen2.pgen import DFAState
import os
import torch
import zipfile
import glob
import pandas as pd
from torch.utils.data import Dataset
from utils.dataset_utils import (read_shp_file,
                                 rgb_tif_tuple,
                                 get_crop, read_fiona)
from torchvision.transforms import Resize, Normalize
from urllib.parse import urlparse
import json
import torch.nn as nn


class FieldDataset(Dataset):
    def __init__(self, label_csv_path, data_dir="data"):
        super().__init__()
        label_csv = pd.read_csv(label_csv_path)
        label_csv = label_csv.set_index("flight_code")
        self.data_dir = data_dir
        image_paths = label_csv['paths']
        mapping = {item: i for i, item in enumerate(label_csv["ActualStatus"].unique())}
        label_csv["Label"] = label_csv["ActualStatus"].apply(lambda x: mapping[x])
        self.collection = []

        for i, d in image_paths.iteritems():
            self.collection.append(
                (
                    {k: os.path.join(
                        self.data_dir, urlparse(v).path[1:]
                    ) for k, v in json.loads(d.replace("'", "\"")).items()},
                    label_csv.loc[i]["Label"]))
        self.resize = Resize((256, 256))
        self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        paths, label = self.collection[index]
        geom = read_shp_file(paths['border_file_path'])
        red = torch.Tensor(get_crop(paths['red_path'], geom))
        blue = torch.Tensor(get_crop(paths['blue_path'], geom))
        green = torch.Tensor(get_crop(paths['green_path'], geom))
        img = torch.cat([red, green, blue], dim=0)
        img = self.normalize(self.resize(img))
        return img, torch.Tensor([label])
