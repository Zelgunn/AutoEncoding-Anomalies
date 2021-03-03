import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import metrics
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context
import numpy as np
import cv2
from time import time
from typing import Tuple, Optional, Callable, Union

from anomaly_detection import IOCompareModel
from callbacks import TensorBoardPlugin
from misc_utils.plot_utils import plot_line2d_to_array
from misc_utils.general import to_constant_list
from misc_utils.numpy_utils import normalize
from datasets import SubsetLoader
from modalities import Pattern


class AUCWrapper(object):
    def __init__(self,
                 curve: str,
                 num_thresholds: int,
                 plot_size: Optional[Tuple[int, int]],
                 prefix=""):
        if curve not in ["ROC", "PR"]:
            raise ValueError("`curve`({}) not in [ROC, PR].".format(curve))

        self.curve = curve
        self.curve_name = curve.lower()
        self.num_thresholds = num_thresholds
        self.plot_size = plot_size
        self.prefix = prefix

        self.auc = metrics.AUC(curve=self.curve, name=self.base_name)

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

    # @tf.function
    def auc_summary(self, y_true, y_pred, step):
        self.reset_states()
        self.update_state(y_true, y_pred)
        auc_scalar = self.result()
        summary_ops_v2.scalar(name=self.base_name + "_scalar", tensor=auc_scalar, step=step)
        return auc_scalar

    def write_plot_summary(self, step):
        if self.plot_size is None:
            return

        fpr = self.false_positive_rate.numpy()
        tpr = self.true_positive_rate.numpy()

        auc_plot_image = plot_line2d_to_array(fpr, tpr, self.plot_size, normalize=False)
        auc_plot_image = cv2.cvtColor(auc_plot_image, cv2.COLOR_RGB2GRAY)
        auc_plot_image = np.expand_dims(auc_plot_image, axis=-1)
        auc_plot_image = np.expand_dims(auc_plot_image, axis=0)

        tf.summary.image(name=self.base_name + "_plot", data=auc_plot_image, step=step)

    def update_state(self, y_true, y_pred):
        self.auc.update_state(y_true, y_pred)

    def reset_states(self):
        self.auc.reset_states()

    def result(self):
        return self.auc.result()

    @property
    def base_name(self):
        if (self.prefix is not None) and (len(self.prefix) > 0):
            return "{}_{}".format(self.prefix, self.curve_name)
        else:
            return self.curve_name


class AUCCallback(TensorBoardPlugin):
    def __init__(self,
                 predictions_model: Model,
                 tensorboard: TensorBoard,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 labels: np.ndarray,
                 update_freq: int or str = "epoch",
                 epoch_freq: int = None,
                 plot_size: Tuple = None,
                 batch_size=32,
                 num_thresholds=100,
                 name="AUC_Callback",
                 prefix=""):
        super(AUCCallback, self).__init__(tensorboard, update_freq, epoch_freq)

        self.predictions_model = predictions_model
        self.plot_size = plot_size
        self.batch_size = batch_size
        self.num_thresholds = num_thresholds
        self.name = name
        self.prefix = prefix

        with tf.name_scope(self.name):
            self.inputs = to_constant_list(inputs, name="input")
            if inputs is outputs:
                self.outputs = self.inputs
            elif outputs is not None:
                self.outputs = to_constant_list(outputs, name="outputs")
            else:
                self.outputs = None

            if labels.shape[-1] == 1:
                if isinstance(labels, np.ndarray):
                    labels = np.squeeze(labels, axis=-1)
                else:
                    labels = tf.squeeze(labels, axis=-1)

            self.labels = tf.constant(labels, name="labels")

            self.roc = AUCWrapper(curve="ROC",
                                  num_thresholds=num_thresholds,
                                  plot_size=self.plot_size,
                                  prefix=prefix)

            self.pr = AUCWrapper(curve="PR",
                                 num_thresholds=num_thresholds,
                                 plot_size=None,
                                 prefix=prefix)

    def _write_logs(self, index):
        start_time = time()

        model_inputs = self.get_model_inputs()
        predictions = self.predictions_model.predict(model_inputs, batch_size=self.batch_size)
        predictions = np.array(predictions)
        predictions = self.reformat_predictions(predictions)
        predictions = normalize(predictions)

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                with self.validation_run_writer.as_default():
                    self._write_auc_summary(predictions, index)
        print("AUCCallback `{}_{}` took {:.2f} seconds.".format(self.prefix, self.name, time() - start_time))

    def get_model_inputs(self):
        if isinstance(self.predictions_model, IOCompareModel):
            model_inputs = self.inputs + self.outputs
        else:
            model_inputs = self.inputs
            if len(model_inputs) == 1:
                model_inputs = model_inputs[0]
        return model_inputs

    def _write_auc_summary(self, predictions, step):
        roc = self.roc.auc_summary(self.labels, predictions, step)
        roc = roc.numpy()

        pr = self.pr.auc_summary(self.labels, predictions, step)
        pr = pr.numpy()

        print("\n==== {}_{} - ROC : {} | PR : {} ====".format(self.prefix, self.name, round(roc, 3), round(pr, 3)))

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
                    predictions = self.reshape_predictions_to_labels(predictions)
            else:
                predictions = self.reshape_predictions_to_labels(predictions)
        return predictions

    def reshape_predictions_to_labels(self, predictions):
        labels_size = np.prod(self.labels.shape)
        if predictions.size == labels_size:
            predictions = np.reshape(predictions, self.labels.shape)
        else:
            # Useful when multiple predictions are made from a single input (e.g. several patches from one image)
            ratio = predictions.size // labels_size
            if int(predictions.size / labels_size) != ratio or ratio < 1:
                raise ValueError("Could not reshape predictions with shape {} to match labels with shape {}"
                                 .format(predictions.shape, self.labels.shape))
            predictions = np.reshape(predictions, [self.labels.shape[0], ratio, *self.labels.shape[1:]])
            predictions = np.mean(predictions, axis=1)

        return predictions

    @staticmethod
    def from_subset(predictions_model: Union[IOCompareModel, Callable],
                    tensorboard: TensorBoard,
                    test_subset: SubsetLoader,
                    pattern: Pattern,
                    labels_length: int,
                    samples_count=512,
                    epoch_freq=1,
                    batch_size=32,
                    prefix="",
                    seed=None
                    ) -> "AUCCallback":
        batch = test_subset.get_batch(batch_size=samples_count, pattern=pattern, seed=seed)

        if pattern.output_count == 2:
            inputs, labels = batch
            outputs = inputs
        elif pattern.output_count == 3:
            inputs, outputs, labels = batch
        else:
            raise ValueError("Pattern's length is {} and should either be 2 and 3.".format(pattern.output_count))

        if not isinstance(predictions_model, IOCompareModel):
            outputs = None

        labels = SubsetLoader.timestamps_labels_to_frame_labels(labels, labels_length)

        auc_callback = AUCCallback(predictions_model=predictions_model,
                                   tensorboard=tensorboard,
                                   inputs=inputs,
                                   outputs=outputs,
                                   labels=labels,
                                   epoch_freq=epoch_freq,
                                   plot_size=(128, 128),
                                   batch_size=batch_size,
                                   prefix=prefix)

        return auc_callback
