import yaml

def parse_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict   