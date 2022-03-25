import yaml


def read_yaml(path):
    """
    YAML reader for config files
    :param path: path to YAML file
    :return: configs
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config
