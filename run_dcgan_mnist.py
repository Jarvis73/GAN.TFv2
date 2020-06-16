import os
import imageio
import tensorflow as tf
import tensorflow.keras as K
from sacred import Experiment
from pathlib import Path
import matplotlib.pyplot as plt

from data_kits import mnist
from networks import dcgan
from utils.constants import NAME, ModeKeys
from utils import loggers
from utils import utils
from utils.timer import Timer

from config import global_ingredient, device_ingredient
from data_kits.mnist import data_ingredient
from networks.dcgan import model_ingredient

ex = Experiment(NAME, ingredients=[
    global_ingredient, device_ingredient, data_ingredient, model_ingredient])


@ex.config_hook
def config_hook(config, command_name, logger):
    _ = logger
    if command_name in ["train"]:
        ex.logger = loggers.get_global_logger(config["g"]["model_dir"], mode=command_name, name=NAME)

    d = config["d"]
    gpus = d["gpus"]
    if isinstance(gpus, int):
        gpus = [gpus]
    if not isinstance(gpus, list):
        raise TypeError(f"Argument `gpus` only support int/list.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus])
    d["gpus"] = gpus

    # Limit gpu memory usage
    all_gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in all_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    return {"d": d}


class Trainer(object):
    def __init__(self, config, model):
        """ Training class
        Parameters
        ----------
        config: ReadOnlyDict, all the configuraitons
        model: dcgan.DCGAN
        """
        self.cfg = config
        self.model = model
        self.generator = self.model.generator
        self.discriminator = self.model.discriminator
        self.timer = Timer()
        self.logger = loggers.get_global_logger(name=NAME)
        self.template_state = "Epoch: {:d}/{:d} - Learning rate: g={:g}, d={:g} - Speed {:.1f} it/s"
        # build solver
        self.generator_optimizer = K.optimizers.Adam(1e-4)
        self.discriminator_optimizer = K.optimizers.Adam(1e-4)
        self.cross_entropy = K.losses.BinaryCrossentropy(from_logits=True)
        self.train_loss_metric = K.metrics.Mean(name="loss")

        self.model_dir = Path(self.cfg.g.model_dir)
        self.ckpt_dir = self.model_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir = self.model_dir / "pred"
        self.pred_dir.mkdir(parents=True, exist_ok=True)

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
        noise = tf.random.normal([self.cfg.data.batch_size, self.cfg.m.noise_dim])

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

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")
        plt.savefig(str(self.pred_dir / f"image_at_epoch_{epoch:04d}"))
        plt.close(fig)

    def start_training_loop(self, train_dataset):
        end_epochs = 50
        d = self.cfg.data
        self.model.generator.build((d.batch_size, self.cfg.m.noise_dim))
        self.model.discriminator.build((d.batch_size, d.height, d.width, d.channel))
        seed = tf.random.normal([self.cfg.m.num_examples_to_generate, self.cfg.m.noise_dim])

        for epoch in range(1, end_epochs + 1):
            for images in train_dataset:
                self.timer.tic()
                self.train_step(images)
                self.timer.toc()

            self.logger.info(self.template_state.format(
                epoch, end_epochs, self.generator_optimizer.lr.numpy(), self.discriminator_optimizer.lr.numpy(),
                self.timer.calls / self.timer.total_time))
            if epoch % 15 == 0:
                self.ckpt.save(str(self.ckpt_dir / "ckpt"))
            self.generate_and_save_images(epoch, seed)


@ex.command
def train(_run, _config):
    cfg = utils.Map(_config)
    utils.set_seed(cfg.seed)
    logger = loggers.get_global_logger(name=NAME)

    # Build Model and Trainer
    logger.info(f"Initialize ==> Model {cfg.m.model}")
    model = dcgan.DCGAN()
    logger.info(f"           ==> Trainer")
    train_obj = Trainer(cfg, model)

    # Build data loader
    logger.info(f"           ==> Data loader for {ModeKeys.TRAIN}")
    train_dataset = mnist.loader(ModeKeys.TRAIN)[0]

    # Start training
    try:
        train_obj.start_training_loop(train_dataset)
    except KeyboardInterrupt:
        logger.info("Main process is terminated by user.")
    finally:
        logger.info(f"Ended running with id {_run._id}.")


@ex.command
def inspect(_run, _config):
    cfg = utils.Map(_config)
    model = dcgan.DCGAN()
    noise = tf.random.normal([1, 100])
    generated_image = model.generator(noise, training=False)
    decision = model.discriminator(generated_image)

    print(decision)
    plt.imshow(generated_image[0, :, :, 0], cmap="gray")
    plt.show()


@ex.command
def animate(_run, _config):
    cfg = utils.Map(_config)
    anim_file = Path(cfg.g.model_dir) / "dcgan.gif"
    pred_dir = Path(cfg.g.model_dir) / "pred"

    with imageio.get_writer(str(anim_file), mode='I') as writer:
        filenames = sorted(pred_dir.glob("image*.png"))
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
    ex.run_commandline()
