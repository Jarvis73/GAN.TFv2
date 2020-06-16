import random
import numpy as np
from pathlib import Path
import tensorflow as tf
from sacred.config.custom_containers import ReadOnlyDict


class Map(ReadOnlyDict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = Map(v)
                else:
                    new_dict[k] = v
        super(Map, self).__init__(new_dict, **kwargs)



def find_checkpoint(checkpoint_dir, latest_filename="checkpoint", checkpoint=None):
    if checkpoint:
        if Path(checkpoint + ".index").exists():
            return checkpoint
        else:
            raise FileNotFoundError(f"Missing checkpoint file in {checkpoint}")
    latest_path = tf.train.latest_checkpoint(checkpoint_dir, latest_filename)
    if not latest_path:
        raise FileNotFoundError(f"Missing checkpoint file in {checkpoint_dir} with status_file {latest_filename}")
    return latest_path


def check_size(shape, *args):
    if isinstance(shape, (list, tuple)):
        return [None if x <= 0 else x for x in shape]
    elif isinstance(shape, (int, float)):
        return [None if shape <= 0 else shape] + [None if x <= 0 else x for x in args]
    else:
        raise TypeError(f"Unsupported type of shape: {type(shape)}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
