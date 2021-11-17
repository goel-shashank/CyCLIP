import os
import io
import yaml
from .utils import *

def read(path):
    config = open(os.path.join(root, path), "r", encoding = "utf-8").read()
    config = config.replace("$DIR", root)
    config = io.StringIO(config)
    config = yaml.load(config, Loader = yaml.FullLoader)
    return config