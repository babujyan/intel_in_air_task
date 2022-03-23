import torch
from utils.general_utils import read_yaml
from models.unet import UNet
from models.classification_model import ClassificationModel

class ModelBuilder:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)
        unet_statedict = torch.load(self.config['unet_pretrained'], map_location=self.config['device'])
        self.unet = UNet(n_channels=3, n_classes=2, bilinear=False)
        self.unet.load_state_dict(unet_statedict)
    
    def build(self):
        return ClassificationModel(self.unet, 6)
