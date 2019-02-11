from keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
from tensorboard.plugins.pr_curve import summary as pr_summary
import cv2
from typing import Tuple

from callbacks import TensorBoardPlugin, CallbackModel
from utils.plot_utils import plot_line2d_to_array


class AUCCallback(TensorBoardPlugin):
    def __init__(self,
                 predictions_model: CallbackModel,
                 tensorboard: TensorBoard,
                 images: np.ndarray,
                 labels: np.ndarray,
                 update_freq: int or str = "epoch",
                 plot_size: Tuple = None,
                 batch_size=32,
                 num_thresholds=100,
                 name="AUC_Callback"):
        super(AUCCallback, self).__init__(tensorboard, update_freq)

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

            self.auc_op = tf.metrics.auc(labels=self.labels_placeholder,
                                         predictions=self.predictions_placeholder,
                                         num_thresholds=num_thresholds)
            self.auc_variables = tf.local_variables(scope=self.name + "/auc")
            self.true_positive_rate = self.auc_variables[0]
            self.false_negative_rate = self.auc_variables[1]
            self.false_positive_rate = self.auc_variables[3]

            self.pr_summary_op = pr_summary.op(name="pr_curve",
                                               labels=self.labels_placeholder,
                                               predictions=self.predictions_placeholder,
                                               num_thresholds=num_thresholds)
        # endregion

        self.auc_image_input = None
        self.auc_image_summary = None
        self.auc_value_input = None
        self.auc_value_summary = None
        self.auc_summary_op = None

    def _write_logs(self, index):
        self.session.run(tf.variables_initializer(self.auc_variables))

        predictions = self.predictions_model.predict(self.images, self.batch_size)
        predictions = self.reformat_predictions(predictions)

        pred_min = predictions.min()
        predictions = (predictions - pred_min) / (predictions.max() - pred_min)

        # region Generate AUC values and PR summary
        results = self.session.run([*self.auc_op, self.pr_summary_op],
                                   feed_dict={self.labels_placeholder: self.labels,
                                              self.predictions_placeholder: predictions})
        _, auc, pr_summary_result = results

        tpr, fnr, fpr = self.session.run([self.true_positive_rate, self.false_negative_rate, self.false_positive_rate])
        # endregion

        # region Plot AUC
        auc_plot_image = plot_line2d_to_array(fpr, tpr, self.plot_size)
        auc_plot_image = cv2.cvtColor(auc_plot_image, cv2.COLOR_RGB2GRAY)
        auc_plot_image = np.expand_dims(auc_plot_image, axis=-1)
        auc_plot_image = np.expand_dims(auc_plot_image, axis=0)
        # endregion

        if self.plot_size is None:
            self.plot_size = tuple(auc_plot_image.shape[1:-1])

        # region Create image/scalar summary ops (if not already created)
        if self.auc_image_input is None:
            self.auc_image_input = tf.placeholder(dtype=tf.float32, shape=[1, *self.plot_size, 1],
                                                  name=self.name + "_image_summary_input")
            self.auc_value_input = tf.placeholder(dtype=tf.float32, shape=[],
                                                  name=self.name + "_value_summary_input")
            self.auc_image_summary = tf.summary.image(name=self.name + "_plot", tensor=self.auc_image_input)
            self.auc_value_summary = tf.summary.scalar(name=self.name + "_scalar", tensor=self.auc_value_input)
            self.auc_summary_op = tf.summary.merge([self.auc_image_summary, self.auc_value_summary])
        # endregion

        auc_summary_result = self.session.run(self.auc_summary_op, feed_dict={self.auc_image_input: auc_plot_image,
                                                                              self.auc_value_input: auc})

        self.tensorboard.writer.add_summary(auc_summary_result, index)
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
