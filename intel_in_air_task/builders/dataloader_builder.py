from intel_in_air_task.utils.general_utils import read_yaml



class DataloaderBuilder:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)
    
    def build(self, mode="train"):
        pass