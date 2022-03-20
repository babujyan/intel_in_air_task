import argparse
from intel_in_air_task.builders.trainer_builder import TrainerBuilder
from intel_in_air_task.builders.model_builder import ModelBuilder
from intel_in_air_task.builders.dataloader_builder import DataloaderBuilder


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        help="path to the config where the model parameters are specified")
    parser.add_argument(
        "--trainer_config",
        help="path to the config where the trainer parameters are specified")
    parser.add_argument(
        "--data_config", 
        help="path to the config where the data/dataloader parameters are specified"
    )
    
    args = parser.parse_args()
    trainer_builder = TrainerBuilder(args.trainer_config)
    model_builder = ModelBuilder(args.model_config)
    dataloader_builder = DataloaderBuilder(args.data_config)


    trainer = trainer_builder.build()
    model = model_builder.build()
    train_dataloader = dataloader_builder.build("train")
    validation_dataloader = dataloader_builder.build("validation")


    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=validation_dataloader)
