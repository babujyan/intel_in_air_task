import argparse
from builders.model_builder import ModelBuilder
from builders.trainer_builder import TrainerBuilder
from utils.dataset_utils import load_image_inference

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
        "--red_path", '-r', type=str, help="path to red .tif file"
    )
    parser.add_argument(
        "--blue_path", '-b', type=str, help="path to blue .tif file"
    )
    parser.add_argument(
        "--green_path", '-g', type=str, help="path to green .tif file"
    )
    parser.add_argument(
        "--boundary_path", type=str, help="path to boundary .zip file"
    )

    args = parser.parse_args()
    model_builder = ModelBuilder(args.model_config)
    model = model_builder.build()
    model_builder.load_ckpt(model)
    img = load_image_inference(args.red_path,
                               args.blue_path,
                               args.green_path,
                               args.boundary_path)
    img = img.unsqueeze(dim=0)
    print(img.shape)
    trainer_builder = TrainerBuilder(args.trainer_config)
    trainer = trainer_builder.build()
    result = trainer.predict(model, img)
    print(result)
