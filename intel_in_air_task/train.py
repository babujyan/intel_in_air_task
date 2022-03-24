import argparse
from builders.trainer_builder import TrainerBuilder
from builders.model_builder import ModelBuilder
from builders.dataloader_builder import DataloaderBuilder
from utils.general_utils import read_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        help="path to the config where the model parameters are specified",
        default="configs/model.yaml"
    )
    parser.add_argument(
        "--trainer_config",
        help="path to the config where the trainer parameters are specified",
        default="configs/trainer.yaml"
    )
    parser.add_argument(
        "--data_config",
        help="path to the config where the data/dataloader "
             "parameters are specified",
        default="configs/dataloader.yaml"
    )

    args = parser.parse_args()
    print(read_yaml(args.trainer_config))
    trainer_builder = TrainerBuilder(args.trainer_config)
    model_builder = ModelBuilder(args.model_config)
    dataloader_builder = DataloaderBuilder(args.data_config)
    trainer = trainer_builder.build()
    model = model_builder.build()
    train_dataloader = dataloader_builder.build("train")
    validation_dataloader = dataloader_builder.build("validation")
    trainer.fit(model, train_dataloader=train_dataloader,
                val_dataloaders=validation_dataloader)
