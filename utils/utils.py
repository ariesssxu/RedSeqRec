from distutils.util import strtobool
import yaml
from addict import Dict

def parse_bool(bool_str):
    return bool(strtobool(bool_str))

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        config = Dict(config)
        return config
