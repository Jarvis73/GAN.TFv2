############################################################################################
#
#  Code: DCGAN on mnist dataset.
#  Date: June 15, 2020
#
#  Reference: https://www.tensorflow.org/tutorials/generative/dcgan
#
############################################################################################

import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
from pathlib import Path

L = K.layers

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


class DCGAN(object):
    def __init__(self, build=True):
        if build:
            self.generator = self.create_generator()
            self.discriminator = self.create_discriminator()
        self.model_dir = Path(__file__).parent / "model_dir/dcgan"
        self.ckpt_dir = self.model_dir / "ckpt"
        self.pred_dir = self.model_dir / "pred"

    def load_data(self):
        (train_images, train_labels), (_, _) = K.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5

        self.train_dataset = (tf.data.Dataset
                              .from_tensor_slices(train_images)
                              .shuffle(BUFFER_SIZE)
                              .batch(BATCH_SIZE))

    @staticmethod
    def create_generator():
        model = K.Sequential([
            L.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)),
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
            L.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]),
            L.LeakyReLU(),
            L.Dropout(0.3),

            L.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            L.LeakyReLU(),
            L.Dropout(0.3),

            L.Flatten(),
            L.Dense(1)
        ])
        return model

    def create_optimizer(self):
        self.cross_entropy = K.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = K.optimizers.Adam(1e-4)
        self.discriminator_optimizer = K.optimizers.Adam(1e-4)

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = tf.train.Checkpoint(generator=self.generator,
                                        discriminator=self.discriminator,
                                        generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer)

    def generator_loss(self, fake):
        return self.cross_entropy(tf.ones_like(fake), fake)

    def discriminator_loss(self, real, fake):
        real_loss = self.cross_entropy(tf.ones_like(real), real)
        fake_loss = self.cross_entropy(tf.zeros_like(fake), fake)
        total_loss = real_loss + fake_loss
        return total_loss


    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_image = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_image, training=True)

            gen_loss = self.generator_loss(fake_output)
            dis_loss = self.discriminator_loss(real_output, fake_output)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        dis_grads = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(dis_grads, self.discriminator.trainable_variables))

    def train(self):
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(EPOCHS):
            start = time.time()

            for batch in self.train_dataset:
                self.train_step(batch)

            self.generate_and_save_images(epoch + 1, seed)

            if (epoch + 1) % 15 == 0:
                self.ckpt.save(str(self.ckpt_dir / "ckpt"))
            print(f"Time for epoch {epoch + 1} is {time.time() - start} sec")
        self.generate_and_save_images(EPOCHS, seed)

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")
        plt.savefig(str(self.pred_dir / f"image_at_epoch_{epoch:04d}"))
        plt.close(fig)

    def inspect():
        noise = tf.random.normal([1, 100])
        generated_image = self.generator(noise, training=False)
        decision = self.discriminator(generated_image)

        print(decision)
        plt.imshow(generated_image[0, :, :, 0], cmap="gray")
        plt.show()

    def animate(self):
        anim_file = self.model_dir / "dcgan.gif"

        with imageio.get_writer(str(anim_file), mode='I') as writer:
            filenames = sorted(self.pred_dir.glob("image*.png"))
            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    model = DCGAN(build=False)
    # model.inspect()

    # model.load_data()
    # model.create_optimizer()
    # model.train()

    model.animate()
