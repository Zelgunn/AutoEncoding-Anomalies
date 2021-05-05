from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging
import numpy as np
from typing import Optional


class CustomModelCheckpoint(Callback):
    def __init__(self,
                 filepath: str,
                 monitor="val_loss",
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode="auto",
                 save_frequency="epoch"
                 ):
        super(CustomModelCheckpoint, self).__init__()

        self.validate_save_frequency(save_frequency)

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_frequency = save_frequency

        self.mode = self.get_mode(mode)
        monitor_op, best = self.get_monitor_op(self.mode, self.monitor)
        self.monitor_op = monitor_op
        self.best = best

        self.model: Optional[Model] = None
        self._chief_worker_only = False
        self._batch_seen_since_last_save = 0
        self._current_epoch = 0

    def save_model(self, logs=None):
        if logs is None:
            logs = {}

        filepath = self.filepath.format(epoch=self._current_epoch, **logs)
        save_now = True
        if self.save_best_only:
            current = logs.get(self.monitor)

            if current is None:
                logging.warning("Can save best model only with available, skipping.".format(self.monitor))
                save_now = False
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print("\nEpoch {epoch:05d}: {monitor} improve from {previous_best:0.5f} to {new_best}. "
                              "Saving model to {filepath}.".
                              format(epoch=self._current_epoch, monitor=self.monitor, previous_best=self.best,
                                     new_best=current, filepath=filepath))
                    self.best = current
                else:
                    save_now = False
                    if self.verbose > 0:
                        print("\nEpoch {epoch:05d}: {monitor} did not improve from {best:0.5f}".
                              format(epoch=self._current_epoch, monitor=self.monitor, best=self.best))

        if save_now:
            self.model.save(filepath=filepath, overwrite=True, include_optimizer=not self.save_weights_only)

    def on_batch_end(self, batch, logs=None):
        if not isinstance(self.save_frequency, int):
            return

        self._batch_seen_since_last_save += 1
        save_now = self._batch_seen_since_last_save == self.save_frequency
        if save_now:
            self.save_model(logs)
            self._batch_seen_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch + 1
        save_now = (self.save_frequency == "epoch") or (epoch % self.save_frequency) == 1
        if save_now:
            self.save_model(logs)

    # region Static helpers
    @staticmethod
    def get_mode(mode: str):
        if mode not in ["auto", "min", "max"]:
            logging.warning("ModelCheckpoint mode {} is unknown, fallback to auto mode.".format(mode))
            mode = "auto"
        return mode

    @staticmethod
    def get_monitor_op(mode: str, monitor):
        if mode == "min":
            monitor_op = np.less
            best = np.Inf
        elif mode == "max":
            monitor_op = np.greater
            best = -np.Inf
        elif ("acc" in monitor) or (monitor.startswith("fmeasure")):
            monitor_op = np.greater
            best = -np.Inf
        else:
            monitor_op = np.less
            best = np.Inf
        return monitor_op, best

    @staticmethod
    def validate_save_frequency(save_frequency):
        if (save_frequency != "epoch") and (not isinstance(save_frequency, int)):
            raise ValueError("Unrecognized save frequency: {}".format(save_frequency))
    # endregion
