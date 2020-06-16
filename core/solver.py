from tensorflow import keras as K
from sacred import Ingredient

train_ingredient = Ingredient("t")      # training


@train_ingredient.config
def training_arguments():
    """ ==> Training Arguments """
    epochs = 0                  # int, Number of epochs for training
    total_epochs = 10           # int,Number of total epochs for training
    ckpt_epoch = 1              # int, Take snapshot every `ckpt_epoch` epochs
    lr = 3e-4                   # float, Base learning rate for model training
    lrp = "none"                # str, Learning rate policy [none/custom_step/period_step/poly/plateau]
    if lrp == "custom_step":
        lr_custom_boundaries = []   # list of int, [custom_step] Use the specified lr at the given boundaries
        lr_custom_values = []       # list of float, [custom_step] Use the specified lr at the given boundaries
    elif lrp == "period_step":
        lr_decay_step = 10          # float, [period_step] Decay the base learning rate at a fixed step
        lr_decay_rate = 0.1         # float, [period_step, plateau] Learning rate decay rate
    elif lrp == "poly":
        lr_power = 0.9              # float, [poly] Polynomial power
        lr_end = 1e-6               # float, [poly, plateau] The minimal end learning rate
    elif lrp == "plateau":
        lr_patience = 30            # int, [plateau] Learning rate patience for decay
        lr_min_delta = 1e-4         # float, [plateau] Minimum delta to indicate improvement
        lr_decay_rate = 0.2         # float, [period_step, plateau] Learning rate decay rate
        lr_end = 1e-6               # float, [poly, plateau] The minimal end learning rate
        cool_down = 0               # int, [plateau]

    optimizer = "adam"          # str, Optimizer for training [adam/momentum]
    if optimizer == "adam":
        beta_1 = 0.9                # float, [adam] Parameter
        beta_2 = 0.99               # float, [adam] Parameter
        epsilon = 1e-7              # float, [adam] Parameter
    elif optimizer == "sgd":
        momentum = 0.9              # float, [momentum] Parameter
        nesterov = False            # bool, [momentum] Parameter


@train_ingredient.capture
def train_epochs(start_epoch,
                 epochs, total_epochs):
    if epochs > 0:
        return epochs + start_epoch
    else:
        return total_epochs


class Solver(object):
    def __init__(self):
        self.lr_callback = self._get_callback()
        self.opt = self._get_model_optimizer()

    @train_ingredient.capture
    def _get_callback(self,
                      lr, lrp, total_epochs=None,
                      lr_custom_boundaries=None, lr_custom_values=None,
                      lr_decay_step=None, lr_decay_rate=None,
                      lr_power=None, lr_end=None,
                      lr_patience=None, lr_min_delta=None, cool_down=None):
        if lrp == "none":
            callback = K.callbacks.LearningRateScheduler(lambda _, lr: lr)
        elif lrp == "plateau":
            callback = K.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                     factor=lr_decay_rate,
                                                     patience=lr_patience,
                                                     mode='min',
                                                     min_delta=lr_min_delta,
                                                     cooldown=cool_down,
                                                     min_lr=lr_end)
        elif lrp == "period_step":
            scheduler = K.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=lr_decay_step,
                decay_rate=lr_decay_rate,
                staircase=True)
            callback = K.callbacks.LearningRateScheduler(scheduler)
        elif lrp == "custom_step":
            scheduler = K.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=lr_custom_boundaries,
                values=lr_custom_values)
            callback = K.callbacks.LearningRateScheduler(scheduler)
        elif lrp == "poly":
            scheduler = K.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=lr,
                decay_steps=total_epochs,
                end_learning_rate=lr_end,
                power=lr_power)
            callback = K.callbacks.LearningRateScheduler(scheduler)
        else:
            raise ValueError('Not supported learning policy.')

        return callback

    @train_ingredient.capture
    def _get_model_optimizer(self,
                             lr, optimizer,
                             beta_1=None, beta_2=None, epsilon=None,
                             momentum=None, nesterov=None):
        if optimizer == "adam":
            optimizer_params = {"beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon}
            optimizer = K.optimizers.Adam(lr, **optimizer_params)
        elif optimizer == "sgd":
            optimizer_params = {"momentum": momentum, "nesterov": nesterov}
            optimizer = K.optimizers.SGD(lr, **optimizer_params)
            optimizer.apply_gradients()
        else:
            raise ValueError("Not supported optimizer: " + optimizer)

        return optimizer