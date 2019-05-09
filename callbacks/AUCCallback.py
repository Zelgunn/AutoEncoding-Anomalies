from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.backend import get_session
from tensorboard.plugins.pr_curve import summary as pr_summary
import cv2
from typing import Tuple, Optional

from callbacks import TensorBoardPlugin
from utils.plot_utils import plot_line2d_to_array


class AUCOperation(object):
    def __init__(self,
                 curve: str,
                 labels: tf.Tensor,
                 predictions: tf.Tensor,
                 num_thresholds: int,
                 plot_size: Optional[Tuple[int, int]]):
        if curve not in ["ROC", "PR"]:
            raise ValueError("`curve`({}) not in [ROC, PR].".format(curve))

        self.curve = curve
        self.curve_name = curve.lower()
        self.labels = labels
        self.predictions = predictions
        self.num_thresholds = num_thresholds
        self.plot_size = plot_size

        self.auc_ops: Tuple[tf.Tensor, tf.Tensor] = tf.metrics.auc(labels=self.labels,
                                                                   predictions=self.predictions,
                                                                   num_thresholds=num_thresholds,
                                                                   curve=self.curve,
                                                                   name=self.curve_name,
                                                                   summation_method="careful_interpolation")

        name_scope: str = self.auc_ops[1].graph.get_name_scope()
        self.auc_variables = tf.local_variables(scope=name_scope + "/{}".format(self.curve_name))

        self.true_positive_rate = self.auc_variables[0]
        self.false_negative_rate = self.auc_variables[1]
        self.false_positive_rate = self.auc_variables[3]

        self.rates = [self.true_positive_rate, self.false_negative_rate, self.false_positive_rate]

        self.initializer = tf.variables_initializer(self.auc_variables)

        self.auc_image_input = None
        self.auc_image_summary = None
        self.auc_value_input = None
        self.auc_value_summary = None
        self.auc_summary_op = None

    def make_auc_summary_ops(self):
        self.auc_value_input = tf.placeholder(dtype=tf.float32, shape=[],
                                              name=self.curve_name + "_value_summary_input")
        self.auc_value_summary = tf.summary.scalar(name=self.curve_name + "_scalar", tensor=self.auc_value_input)

        if self.plot_size is not None:
            self.auc_image_input = tf.placeholder(dtype=tf.float32, shape=[1, *self.plot_size, 1],
                                                  name=self.curve_name + "_image_summary_input")
            self.auc_image_summary = tf.summary.image(name=self.curve_name + "_plot", tensor=self.auc_image_input)

            self.auc_summary_op = tf.summary.merge([self.auc_image_summary, self.auc_value_summary])
        else:
            self.auc_summary_op = self.auc_value_summary

    def run_summary(self, auc: float, auc_plot_image: Optional[np.ndarray] = None):
        if self.auc_summary_op is None:
            self.make_auc_summary_ops()

        feed_dict = {self.auc_value_input: auc}
        if self.plot_size is not None:
            feed_dict[self.auc_image_input] = auc_plot_image

        session = get_session()
        auc_summary_result = session.run(self.auc_summary_op, feed_dict=feed_dict)
        return auc_summary_result


class AUCCallback(TensorBoardPlugin):
    def __init__(self,
                 predictions_model: KerasModel,
                 tensorboard: TensorBoard,
                 images: np.ndarray,
                 labels: np.ndarray,
                 update_freq: int or str = "epoch",
                 epoch_freq: int = None,
                 plot_size: Tuple = None,
                 batch_size=32,
                 num_thresholds=100,
                 name="AUC_Callback"):
        super(AUCCallback, self).__init__(tensorboard, update_freq, epoch_freq)

        self.predictions_model = predictions_model
        self.images = images
        self.labels = np.squeeze(labels)
        self.plot_size = plot_size
        self.batch_size = batch_size
        self.name = name

        # region Create AUC metric + PR summary graphs
        with tf.name_scope(self.name):
            labels_shape = [None, *self.labels.shape[1:]]
            self.labels_placeholder = tf.placeholder(dtype=tf.bool, shape=labels_shape,
                                                     name="auc_labels_placeholder")
            self.predictions_placeholder = tf.placeholder(dtype=tf.float32, shape=labels_shape,
                                                          name="auc_predictions_placeholder")

            self.roc_operation = AUCOperation(curve="ROC",
                                              labels=self.labels_placeholder,
                                              predictions=self.predictions_placeholder,
                                              num_thresholds=num_thresholds,
                                              plot_size=self.plot_size)

            self.pr_operation = AUCOperation(curve="PR",
                                             labels=self.labels_placeholder,
                                             predictions=self.predictions_placeholder,
                                             num_thresholds=num_thresholds,
                                             plot_size=None)

            self.pr_summary_op = pr_summary.op(name="pr_curve",
                                               labels=self.labels_placeholder,
                                               predictions=self.predictions_placeholder,
                                               num_thresholds=num_thresholds)
        # endregion

    def _write_logs(self, index):
        # region Initialize ROC/PR local variables
        self.session.run(self.roc_operation.initializer)
        self.session.run(self.pr_operation.initializer)
        # endregion

        # region Make predictions
        predictions = self.predictions_model.predict(self.images, self.batch_size)
        predictions = np.array(predictions)
        predictions = self.reformat_predictions(predictions)

        pred_min = predictions.min()
        predictions = (predictions - pred_min) / (predictions.max() - pred_min)
        # endregion

        # region Generate AUC values and PR summary
        auc_ops = self.roc_operation.auc_ops + self.pr_operation.auc_ops
        results = self.session.run([self.pr_summary_op, *auc_ops],
                                   feed_dict={self.labels_placeholder: self.labels,
                                              self.predictions_placeholder: predictions})
        pr_summary_result, *auc_results = results
        _, roc_auc, _, pr_auc = auc_results

        tpr, fnr, fpr = self.session.run(self.roc_operation.rates)
        # endregion

        # region Plot AUC
        roc_auc_plot_image = plot_line2d_to_array(fpr, tpr, self.plot_size)
        roc_auc_plot_image = cv2.cvtColor(roc_auc_plot_image, cv2.COLOR_RGB2GRAY)
        roc_auc_plot_image = np.expand_dims(roc_auc_plot_image, axis=-1)
        roc_auc_plot_image = np.expand_dims(roc_auc_plot_image, axis=0)
        # endregion

        if self.plot_size is None:
            self.plot_size = tuple(roc_auc_plot_image.shape[1:-1])

        roc_auc_summary_result = self.roc_operation.run_summary(roc_auc, roc_auc_plot_image)
        pr_auc_summary_result = self.pr_operation.run_summary(pr_auc)

        self.tensorboard.writer.add_summary(roc_auc_summary_result, index)
        self.tensorboard.writer.add_summary(pr_auc_summary_result, index)
        self.tensorboard.writer.add_summary(pr_summary_result, index)

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
