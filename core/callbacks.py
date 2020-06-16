import os
import json
import time
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import multi_worker_training_state as training_state
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import checkpoint_management

from utils.loggers import logger as logging


class ModelCheckpoint(callbacks.Callback):
    """Save the model after every epoch.
    Modification:
    * Remove some codes
    * Replace model.save/model.save_weights to tf.train.Checkpoint and tf.train.CheckpointManager
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`, the latest best model according
          to the quantity monitored will not be overwritten.
          If `filepath` doesn't contain formatting options like `{epoch}` then
          `filepath` will be overwritten by each new better model.
        mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
          overwrite the current save file is made based on either the maximization
          or the minimization of the monitored quantity. For `val_acc`, this
          should be `max`, for `val_loss` this should be `min`, etc. In `auto`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        save_weights_only: if True, then only the model's weights will be saved
          (`model.save_weights(filepath)`), else the full model is saved
          (`model.save(filepath)`).
        save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
          the model after each epoch. When using integer, the callback saves the
          model at end of a batch at which this many samples have been seen since
          last saving. Note that if the saving isn't aligned to epochs, the
          monitored metric may potentially be less reliable (it could reflect as
          little as 1 batch, since the metrics get reset every epoch). Defaults to
          `'epoch'`
        **kwargs: Additional arguments for backwards compatibility. Possible key
          is `period`.
    """

    def __init__(self,
                 filepath,
                 directory,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 mode='min',
                 save_freq='epoch',
                 max_to_keep=5):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = os.path.join(directory, filepath)
        self.directory = directory
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.epochs_since_last_save = 0
        self._samples_seen_since_last_saving = 0
        self.period = 1

        if mode not in ['min', 'max']:
            raise ValueError('ModelCheckpoint mode %s is unknown, fallback to auto mode.' % mode)
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf

        if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
            raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False

    def set_params(self, params):
        self.start_epoch = params

    def set_model(self, model):
        self.model = model
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1, trainable=False),
                                        optimizer=model.optimizer, model=model)
        self.manager = CheckpointManagerV2(self.ckpt, self.directory, max_to_keep=self.max_to_keep)

    def on_train_begin(self, logs=None):
        # pylint: disable=protected-access
        if self.model._in_multi_worker_mode():
            # MultiWorkerTrainingState is used to manage the training state needed
            # for preemption-recovery of a worker in multi-worker training.
            self.model._training_state = (
                training_state.MultiWorkerTrainingState(self.model, self.filepath))
            self._training_state = self.model._training_state
            if self._training_state.restore():
                # If the training state needs to be and is successfully restored,
                # it is recovering from a previous failure (or preemption). In such
                # case, do not load the weights from user specified file path.
                return

        # If this is not multi worker training, restoring is not needed, or
        # restoring failed, check if it should load weights on restart.
        if (not self.model._in_multi_worker_mode() or
                multi_worker_util.should_load_checkpoint()):
            filepath_to_load = self.manager.latest_checkpoint
            if (filepath_to_load is not None and
                    training_state.checkpoint_exists(filepath_to_load)):
                try:
                    # `filepath` may contain placeholders such as `{epoch:02d}`, and
                    # thus it attempts to load the most recently modified file with file
                    # name matching the pattern.
                    if not self.save_best_only:
                        self.ckpt.restore(filepath_to_load).expect_partial()
                        self.start_epoch[0] = self.ckpt.step.numpy() + 1
                        logging.info(f"Restored checkpoint from {filepath_to_load}. "
                                     f"Start from epoch {self.start_epoch[0]}.")
                    else:
                        # Try to restore best metric
                        self._load_metric(filepath_to_load)
                except (IOError, ValueError) as e:
                    raise ValueError('Error loading file from {}. Reason: {}'.format(
                        filepath_to_load, e))

    def on_train_end(self, logs=None):
        # pylint: disable=protected-access
        if self.model._in_multi_worker_mode():
            if self.model.stop_training or getattr(
                    self.model, '_successful_loop_finish', False):
                # In multi-worker training, on successful exit of training, delete the
                # training state backup file that was saved for the purpose of worker
                # recovery.
                self._training_state.delete_backup()
                # Restore the training state so the model is ready for next (possible)
                # multi worker training.
                del self._training_state
                del self.model._training_state

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if isinstance(self.save_freq, int):
            self._samples_seen_since_last_saving += logs.get('size', 1)
            if self._samples_seen_since_last_saving >= self.save_freq:
                self._save_model(epoch=self._current_epoch, logs=logs)
                self._samples_seen_since_last_saving = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == 'epoch':
            if self.model._in_multi_worker_mode():
                # Exclude training state variables in user-requested checkpoint file.
                with self._training_state.untrack_vars():
                    self._save_model(epoch=epoch, logs=logs)
            else:
                self._save_model(epoch=epoch, logs=logs)
        if self.model._in_multi_worker_mode():
            # For multi-worker training, back up the weights and current training
            # state for possible future recovery.
            self._training_state.back_up(epoch)

    def _save_model(self, epoch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                current = logs.get(self.monitor)
                if self.save_best_only:
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    elif self.monitor_op(current, self.best):
                        self.ckpt.step.assign(epoch)
                        self.manager.save_v2(filepath)
                        self._save_metric(filepath, current)
                        if self.verbose > 0:
                            logging.info('    Saved (best) checkpoint to %s. %s improved from %0.5f to %0.5f'
                                         % (filepath, self.monitor, self.best, current))
                        self.best = current
                else:
                    self.ckpt.step.assign(epoch)
                    self.manager.save_v2(filepath)
                    if self.verbose > 0:
                        logging.info('    Saved checkpoint model to %s' % filepath)

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.message`.
                if 'is a directory' in e.message:
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))

    def _save_metric(self, filepath, value):
        filepath = os.path.join(os.path.dirname(filepath), "best_metric")
        metric = {self.monitor: float(value)}
        with open(filepath, "w") as f:
            json.dump(metric, f)

    def _load_metric(self, filepath):
        filepath = os.path.join(os.path.dirname(filepath), "best_metric")
        with open(filepath, "r") as f:
            metric = json.load(f)
        if self.monitor in metric:
            self.best = metric[self.monitor]

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        if not self.model._in_multi_worker_mode(
        ) or multi_worker_util.should_save_checkpoint():
            return self.filepath.format(epoch=epoch, **logs)
        else:
            # If this is multi-worker training, and this worker should not
            # save checkpoint, we use a temp filepath to store a dummy checkpoint, so
            # it writes to a file that will be removed at the end of `_save_model()`
            # call. This is because the SyncOnReadVariable needs to be synced across
            # all the workers in order to be read, and all workers need to initiate
            # that.
            self._temp_file_dir = tempfile.mkdtemp()
            extension = os.path.splitext(self.filepath)[1]
            return os.path.join(self._temp_file_dir, 'temp' + extension)

    def _maybe_remove_file(self):
        # Remove the checkpoint directory in multi-worker training where this worker
        # should not checkpoint. It is a dummy directory previously saved for sync
        # distributed training.

        if (self.model._in_multi_worker_mode() and  # pylint: disable=protected-access
                not multi_worker_util.should_save_checkpoint()):
            file_io.delete_recursively(self._temp_file_dir)
            del self._temp_file_dir


class CheckpointManagerV2(checkpoint_management.CheckpointManager):
    def save_v2(self, filepath):
        # Save counter logic duplicated from tf.train.Checkpoint, soon to diverge
        # slightly with a custom numbering option.
        if context.executing_eagerly():
            save_counter = self._checkpoint.save_counter
            save_counter.assign_add(1)
            session = None
        else:
            session = ops.get_default_session()

            def _initializing_creator(next_creator, **kwargs):
                """Initialize the save counter if it has been newly created."""
                v = next_creator(**kwargs)
                session.run(v.initializer)
                return v

            with variable_scope.variable_creator_scope(_initializing_creator):
                save_counter = self._checkpoint.save_counter
            if self._save_counter_assign is None:
                self._save_counter_assign = save_counter.assign_add(1, read_value=False)
            session.run(self._save_counter_assign)
        save_path = self._checkpoint.write(filepath)
        timestamp = time.time()
        # If this is an overwritten checkpoint we were previously tracking, delete
        # and reinsert it to make sure it goes to the end of the queue.
        if save_path in self._maybe_delete:
            del self._maybe_delete[save_path]
        self._maybe_delete[save_path] = timestamp
        self._latest_checkpoint = save_path
        # Before deleting anything we update the Checkpoint proto with the new
        # checkpoint. We'll go back and correct it after cleaning up old files, but
        # a preemption while deleting will be more likely to see the new checkpoint
        # this way.
        self._record_state()
        self._sweep()
        # Write out the Checkpoint proto a second time, now without the deleted
        # checkpoints.
        self._record_state()
        return save_path
