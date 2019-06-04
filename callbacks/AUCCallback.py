from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from typing import Tuple, Optional

from callbacks import TensorBoardPlugin
from utils.plot_utils import plot_line2d_to_array


class AUCWrapper(object):
    def __init__(self,
                 curve: str,
                 num_thresholds: int,
                 plot_size: Optional[Tuple[int, int]]):
        if curve not in ["ROC", "PR"]:
            raise ValueError("`curve`({}) not in [ROC, PR].".format(curve))

        self.curve = curve
        self.curve_name = curve.lower()
        self.num_thresholds = num_thresholds
        self.plot_size = plot_size

        self.auc = keras.metrics.AUC(curve=self.curve, name=self.curve_name)

    # region Properties
    @property
    def auc_variables(self):
        return self.auc.variables

    @property
    def true_positive_rate(self):
        return self.auc_variables[0]

    @property
    def false_negative_rate(self):
        return self.auc_variables[1]

    @property
    def true_negative_rate(self):
        return self.auc_variables[2]

    @property
    def false_positive_rate(self):
        return self.auc_variables[3]

    # endregion

    @tf.function
    def auc_summary(self, y_true, y_pred, step):
        self.reset_states()
        self.update_state(y_true, y_pred)
        auc_scalar = self.result()
        tf.summary.scalar(name=self.curve_name + "_scalar", data=auc_scalar, step=step)

    def write_plot_summary(self, step):
        if self.plot_size is None:
            return

        fpr = self.false_positive_rate.numpy()
        tpr = self.true_positive_rate.numpy()

        auc_plot_image = plot_line2d_to_array(fpr, tpr, self.plot_size)
        auc_plot_image = cv2.cvtColor(auc_plot_image, cv2.COLOR_RGB2GRAY)
        auc_plot_image = np.expand_dims(auc_plot_image, axis=-1)
        auc_plot_image = np.expand_dims(auc_plot_image, axis=0)

        tf.summary.image(name=self.curve_name + "_plot", data=auc_plot_image, step=step)

    def update_state(self, y_true, y_pred):
        self.auc.update_state(y_true, y_pred)

    def reset_states(self):
        self.auc.reset_states()

    def result(self):
        return self.auc.result()


class AUCCallback(TensorBoardPlugin):
    def __init__(self,
                 predictions_model: keras.Model,
                 tensorboard: TensorBoard,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 labels: np.ndarray,
                 update_freq: int or str = "epoch",
                 epoch_freq: int = None,
                 plot_size: Tuple = None,
                 batch_size=32,
                 num_thresholds=100,
                 name="AUC_Callback"):
        super(AUCCallback, self).__init__(tensorboard, update_freq, epoch_freq)

        self.predictions_model = predictions_model
        self.plot_size = plot_size
        self.batch_size = batch_size
        self.num_thresholds = num_thresholds
        self.name = name

        with tf.name_scope(self.name):
            self.inputs = tf.constant(inputs, name="inputs")
            self.outputs = tf.constant(outputs, name="outputs")
            self.labels = tf.constant(np.squeeze(labels), name="labels")

            self.roc = AUCWrapper(curve="ROC",
                                  num_thresholds=num_thresholds,
                                  plot_size=self.plot_size)

            self.pr = AUCWrapper(curve="PR",
                                 num_thresholds=num_thresholds,
                                 plot_size=None)

    def _write_logs(self, index):
        predictions = self.predictions_model.predict((self.inputs, self.outputs), batch_size=self.batch_size)
        predictions = np.array(predictions)
        predictions = self.reformat_predictions(predictions)

        pred_min = predictions.min()
        predictions = (predictions - pred_min) / (predictions.max() - pred_min)

        with self.validation_run_writer.as_default():
            self._write_auc_summary(predictions, index)
            self.roc.write_plot_summary(index)

    @tf.function
    def _write_auc_summary(self, predictions, step):
        self.roc.auc_summary(self.labels, predictions, step)
        self.pr.auc_summary(self.labels, predictions, step)

    def reformat_predictions(self, predictions):
        if predictions.shape != self.labels.shape:
            if len(self.labels.shape) > 2:
                pred_dim = np.prod(predictions.shape)
                labels_dim = np.prod(self.labels.shape)
                if pred_dim != labels_dim:
                    resized_predictions = np.empty(shape=self.labels.shape, dtype=predictions.dtype)
                    dsize = tuple(reversed(resized_predictions.shape[1:3]))
                    for j in range(len(predictions)):
                        resized_predictions[j] = cv2.resize(predictions[j], dsize, interpolation=cv2.INTER_AREA)
                    predictions = resized_predictions
                else:
                    predictions = np.reshape(predictions, self.labels.shape)
            else:
                predictions = np.reshape(predictions, self.labels.shape)
        return predictions
