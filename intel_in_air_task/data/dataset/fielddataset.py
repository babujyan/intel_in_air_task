import os
import json
import torch
import random
import pandas as pd
from urllib.parse import urlparse
from torch.utils.data import Dataset
from utils.dataset_utils import read_shp_file
from utils.dataset_utils import get_crop
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.transforms import RandomApply
from torchvision.transforms import RandomCrop
from torchvision.transforms import Compose
from torchvision.transforms import ColorJitter
from data.data_label_mapping import mapping


class FieldDataset(Dataset):
    def __init__(self, label_csv_path,
                 data_dir="data", validation_size=0.2, mode='train'):
        """
        For image transformation and augmentations
        :param label_csv_path:
        :param data_dir:
        :param validation_size:
        :param mode:
        """
        super().__init__()
        label_csv = pd.read_csv(label_csv_path)
        if "mode" not in label_csv:
            label_csv['mode'] = \
                random.choices(
                    ['train', 'validation'],
                    weights=[1 - validation_size, validation_size],
                    k=label_csv.shape[0])
        label_csv = label_csv[label_csv['mode'] == mode]
        label_csv = label_csv.set_index("flight_code")
        self.data_dir = data_dir
        image_paths = label_csv['paths']
        label_csv["Label"] = label_csv["ActualStatus"].apply(
            lambda x: mapping[x])
        self.collection = []

        for i, d in image_paths.iteritems():
            self.collection.append(
                (
                    {k: os.path.join(
                        self.data_dir, urlparse(v).path[1:]
                    ) for k, v in json.loads(d.replace("'", "\"")).items()},
                    label_csv.loc[i]["Label"]))

        # Random cropping the image
        self.crop = RandomCrop(128)
        self.jitter = ColorJitter(0.5, 0.5, 0.2, 0.4)
        self.composed = Compose([RandomCrop(128),
                                 ColorJitter(0.5, 0.5, 0.5, 0.5)])
        self.resize = Resize((256, 256))
        self.normalize = Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))
        self.augmentations = RandomApply(
            [self.crop, self.jitter, self.composed], p=0.3
        )

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        paths, label = self.collection[index]
        geom = read_shp_file(paths['border_file_path'])
        red = torch.Tensor(get_crop(paths['red_path'], geom))
        blue = torch.Tensor(get_crop(paths['blue_path'], geom))
        green = torch.Tensor(get_crop(paths['green_path'], geom))
        img = torch.cat([red, green, blue], dim=0)
        img = self.resize(img)
        img = self.augmentations(img)
        img = self.normalize(self.resize(img))
        return img, torch.Tensor([label])
