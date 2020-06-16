import tensorflow as tf
from sacred import Ingredient

from utils.constants import ModeKeys

_data_cache = None
data_ingredient = Ingredient("data")


@data_ingredient.config
def data_arguments():
    """ ==> Data Arguments """
    height = 28                     # int, Image height
    width = 28                      # int, Image width
    channel = 1                     # int, Image channel
    batch_size = 256                # int, Batch size
    buffer_size = 60000             # int, Buffer size for shuffle


@data_ingredient.capture
def loader(mode,
           batch_size, buffer_size):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if isinstance(mode, str):
        mode = [mode]
    elif isinstance(mode, (tuple, list)):
        pass
    else:
        raise TypeError(f"`mode` must have type of str or tuple/list, got `{type(mode)}`")

    all_datasets = []
    if ModeKeys.TRAIN in mode:
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5

        dataset = (tf.data.Dataset
                   .from_tensor_slices(train_images)
                   .shuffle(buffer_size)
                   .batch(batch_size))
        all_datasets.append(dataset)
    else:
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        test_images = (test_images - 127.5) / 127.5

        dataset = (tf.data.Dataset
                   .from_tensor_slices(test_images)
                   .batch(batch_size))
        all_datasets.append(dataset)

    return tuple(all_datasets)
