import torch
from pytorch_lightning import Trainer
from intel_in_air_task.utils.general_utils import read_yaml


class TrainerBuilder:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)
    
    def get_logger(self):
        pass


    def get_callbacks(self):
        pass

    def build(self):
        logger = self.get_logger()
        callbacks = self.get_callbacks()
        trainer = Trainer(**self.config,
                          logger=logger,
                          callbacks=callbacks,
                          )
        return trainer