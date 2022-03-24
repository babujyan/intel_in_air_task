from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.general_utils import read_yaml


class TrainerBuilder:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)
        self.logger_dir = self.config['trainer'].pop('logger_dir')

    def get_logger(self):
        return TensorBoardLogger(save_dir=self.logger_dir,
                                 name='',
                                 version='')

    def get_callbacks(self):
        return [ModelCheckpoint(**self.config['callback'])]

    def build(self):
        logger = self.get_logger()
        callbacks = self.get_callbacks()
        trainer = Trainer(
            **self.config['trainer'],
            weights_summary=None,
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=callbacks
        )
        return trainer
