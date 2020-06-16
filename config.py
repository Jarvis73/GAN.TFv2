import glob
import itertools
from pathlib import Path
from sacred import Ingredient
from sacred.utils import apply_backspaces_and_linefeeds

ROOT = Path(__file__).parent
global_ingredient = Ingredient("g")     # global
device_ingredient = Ingredient("d")     # device


@global_ingredient.config
def global_arguments():
    """ ==> Global Arguments """
    mode = "train"              # str, Model mode for train/val/test/export
    tag = "default"             # str, Configuration tag
    model_dir = "model_dir"     # str/null, Directory to save model parameters, graph, etc
    log_file = ""               # str/null, Logging file name to replace default
    enable_function = True      # bool, Enable tf.function for performance
    if mode == "train":
        log_step = 500              # int, Log running information per `log_step`
        summary_step = log_step     # int, Summary user-defined items to event file per `summary_step`
        summary_prefix = tag        # str, A string that will be prepend to the summary tags
        ckpt_freq = 1               # int, Frequency to take snapshot of the model weights. 0 denotes no snapshots.
    enable_tensorboard = False      # bool, Enable Tensorboard or not


@device_ingredient.config
def device_arguments():
    """ ==> Device Arguments """
    device_mem_frac = 0.            # Used for per_process_gpu_memory_fraction
    distribution_strategy = "off"   # === Don't use! === A string specify which distribution strategy to use
    gpus = 0                        # Which gpu to run this model. For multiple gpus: gpus=[0, 1]
    all_reduce_alg = ""             # === Don't use! === Specify which algorithm to use when performing all-reduce


@global_ingredient.config_hook
def global_args_preprocess(config, command_name, logger):
    _ = command_name
    _ = logger
    model_dir = config["g"]["model_dir"]
    tag = str(config["g"]["tag"])

    if config["g"]["mode"] not in ["train", "eval", "test", "export"]:
        raise ValueError

    if not model_dir:
        model_dir = "model_dir"
    model_dir = ROOT / model_dir
    if not model_dir.exists():
        model_dir.mkdir(exist_ok=True, parents=True)
    new_config = {"model_dir": str(model_dir / tag)}
    return new_config
