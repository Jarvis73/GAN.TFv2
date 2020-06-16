import importlib
from sacred import Ingredient
from tensorflow import keras as K
import tensorflow_addons as tfa

model_ingredient = Ingredient("m")      # model


@model_ingredient.config
def model_arguments():
    """ ==> Model Arguments """
    num_classes = 2                     # int, Number of classes
    weight_init = "glorot_uniform"      # str, Weights initializer. See https://www.tensorflow.org/api_docs/python/tf/keras/initializers
    normalizer = "batch"                # str, Normalizer. [instance_norm/batch_norm]
    weight_regu = "l2"                  # str, Model weights regularizer. [l1/l2]
    weight_decay = 3e-5                 # float, Weight decay for regularizer
    activation = "relu"                 # str, Activation function
    summary = False                     # bool, Summary keras model

    normalizer_params = {
        "center": True,
        "scale": True,
        "epsilon": 1e-3
    }
    if normalizer == "group":
        normalizer_params["group"] = 2

    if activation == "leaky_relu":
        activation_params = {
            "alpha": 0.3
        }


@model_ingredient.capture
def get_kernel_regu(weight_regu, weight_decay):
    if weight_regu == "l2":
        return K.regularizers.l2(weight_decay)
    elif weight_regu == "l1":
        return K.regularizers.l1(weight_decay)
    else:
        raise ValueError(f"weight_regu: {weight_regu}. [l1/l2]")


@model_ingredient.capture
def get_normalizer(normalizer, normalizer_params=None, name=None):
    if normalizer_params is None:
        normalizer_params = {}
    if normalizer == "batch":
        return K.layers.BatchNormalization(name=name, **normalizer_params)
    elif normalizer == "instance":
        return tfa.layers.InstanceNormalization(name=name, **normalizer_params)
    elif normalizer == "layer":
        return K.layers.LayerNormalization(name=name, **normalizer_params)
    elif normalizer == "group":
        return tfa.layers.GroupNormalization(name=name, **normalizer_params)
    else:
        raise ValueError(f"Normalizer: {normalizer}. [batch/instance/layer/group]")


@model_ingredient.capture
def get_activation(activation, activation_params=None, name=None):
    if activation == "relu":
        return K.layers.ReLU(name=name)
    elif activation == "leaky_relu":
        if activation_params is None:
            activation_params = {}
        return K.layers.LeakyReLU(name=name, **activation_params)
    elif activation == "prelu":
        return K.layers.PReLU(name=name)
    else:
        raise ValueError(f"Activation: {activation}. [relu/leaky_relu/prelu]")


def get_model_class_by_name(name):
    file_name = "networks." + name.lower()
    libs = importlib.import_module(file_name)
    if "ModelClass" in libs.__dict__:
        return libs.__dict__["ModelClass"]
    else:
        raise ValueError(f"Missing 'ModelClass' in {file_name}.")