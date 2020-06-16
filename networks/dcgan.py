############################################################################################
#
#  Code: DCGAN on mnist dataset.
#  Date: June 15, 2020
#
#  Reference: https://www.tensorflow.org/tutorials/generative/dcgan
#
############################################################################################

import tensorflow.keras as K
from sacred import Ingredient

L = K.layers

model_ingredient = Ingredient("m")      # model


@model_ingredient.config
def model_arguments():
    """ ==> Model Arguments """
    noise_dim = 100
    num_examples_to_generate = 16


class DCGAN(object):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

    @staticmethod
    def create_generator():
        model = K.Sequential([
            L.Dense(7 * 7 * 256, use_bias=False),
            L.BatchNormalization(),
            L.LeakyReLU(),

            L.Reshape((7, 7, 256)),
            L.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),     # (None, 7, 7, 128)
            L.BatchNormalization(),
            L.LeakyReLU(),

            L.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),      # (None, 14, 14, 64)
            L.BatchNormalization(),
            L.LeakyReLU(),

            L.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                              activation="tanh")                                                # (None, 28, 28, 1)
        ])
        return model

    @staticmethod
    def create_discriminator():
        model = K.Sequential([
            L.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
            L.LeakyReLU(),
            L.Dropout(0.3),

            L.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            L.LeakyReLU(),
            L.Dropout(0.3),

            L.Flatten(),
            L.Dense(1)
        ])
        return model
