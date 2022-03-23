from intel_in_air_task.data.dataset.fielddataset import FieldDataset
from intel_in_air_task.builders.model_builder import ModelBuilder
from intel_in_air_task.builders.dataloader_builder import DataloaderBuilder
dataset = FieldDataset("/home/mane/Desktop/field_state_classification/final_data.CSV", "/home/mane/Desktop/")

sample = dataset.__getitem__(2)
print(len(sample))
print(sample[0].shape)

s = sample[0].unsqueeze(dim=0)
print(s.shape)
mb = ModelBuilder(config_path='configs/model.yaml')
model = mb.build()
output = model.forward(s)
print(output.shape)


dl = DataloaderBuilder('configs/dataloader.yaml')
dl.build()