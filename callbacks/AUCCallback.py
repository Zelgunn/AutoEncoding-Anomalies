from keras.callbacks import Callback, TensorBoard
import numpy as np
import tensorflow as tf
from tensorboard.plugins.pr_curve import summary as pr_summary
import cv2
from typing import Tuple

from callbacks import TensorBoardPlugin
from models import AutoEncoderBaseModel
from scheme import Dataset
from plot_utils import plot_line2d_to_array


class AUCCallback(TensorBoardPlugin):
    def __init__(self,
                 tensorboard: TensorBoard,
                 autoencoder: AutoEncoderBaseModel,
                 dataset: Dataset,
                 update_freq: int or str = "epoch",
                 scale: int = None,
                 plot_size: Tuple = None,
                 batch_size=32,
                 num_thresholds=100):
        super(AUCCallback, self).__init__(tensorboard, update_freq)

        self.autoencoder = autoencoder
        self.images = dataset.images
        self.frame_level_labels = dataset.frame_level_labels()
        self.plot_size = plot_size
        self.batch_size = batch_size

        self.autoencoder_model = self.autoencoder.get_model_at_scale(scale)

        with tf.name_scope("AUC_Callback"):
            self.labels_placeholder = tf.placeholder(dtype=tf.bool, shape=[None],
                                                     name="auc_labels_placeholder")
            self.predictions_placeholder = tf.placeholder(dtype=tf.float32, shape=[None],
                                                          name="auc_predictions_placeholder")

            self.predictions = self.autoencoder.frame_level_average_error(scale, normalize_error=True)

            self.auc_op = tf.metrics.auc(labels=self.labels_placeholder,
                                         predictions=self.predictions_placeholder,
                                         num_thresholds=num_thresholds)
            self.auc_variables = tf.local_variables(scope="AUC_Callback/auc")
            self.true_positive_rate = self.auc_variables[0]
            self.false_negative_rate = self.auc_variables[1]
            self.false_positive_rate = self.auc_variables[3]

            self.pr_summary_op = pr_summary.op(name="pr_curve",
                                               labels=self.labels_placeholder,
                                               predictions=self.predictions_placeholder,
                                               num_thresholds=num_thresholds)

        self.batch_count = int(np.ceil(self.images.shape[0] / self.batch_size))
        self.auc_feed_dicts = []
        for i in range(self.batch_count):
            batch = {self.autoencoder_model.input: self.images[i * batch_size: (i + 1) * batch_size]}
            self.auc_feed_dicts.append(batch)

        self.auc_image_input = None
        self.auc_image_summary = None
        self.auc_value_input = None
        self.auc_value_summary = None
        self.auc_summary_op = None

    def _write_logs(self, index):
        self.session.run(tf.variables_initializer(self.auc_variables))

        predictions = []
        for i in range(self.batch_count):
            batch_predictions = self.session.run(self.predictions, feed_dict=self.auc_feed_dicts[i])
            predictions.append(batch_predictions)
        predictions = np.concatenate(predictions, axis=0)

        results = self.session.run([*self.auc_op, self.pr_summary_op],
                                   feed_dict={self.labels_placeholder: self.frame_level_labels,
                                              self.predictions_placeholder: predictions})
        _, auc, pr_summary_result = results

        tpr, fnr, fpr = self.session.run([self.true_positive_rate, self.false_negative_rate, self.false_positive_rate])

        auc_plot_image = plot_line2d_to_array(fpr, tpr, self.plot_size)
        auc_plot_image = cv2.cvtColor(auc_plot_image, cv2.COLOR_RGB2GRAY)

        rp_plot_x = tpr / (tpr + fnr + 1e-7)
        rp_plot_y = tpr / (tpr + fpr + 1e-7)
        rp_plot_image = plot_line2d_to_array(rp_plot_x, rp_plot_y, self.plot_size, normalize=False)
        rp_plot_image = cv2.cvtColor(rp_plot_image, cv2.COLOR_RGB2GRAY)

        plot_images = [auc_plot_image, rp_plot_image]
        plot_images = np.expand_dims(plot_images, axis=-1)

        if self.plot_size is None:
            self.plot_size = tuple(plot_images.shape[1:-1])

        if self.auc_image_input is None:
            self.auc_image_input = tf.placeholder(dtype=tf.float32, shape=[2, *self.plot_size, 1],
                                                  name="AUCCallback_image_summary_input")
            self.auc_value_input = tf.placeholder(dtype=tf.float32, shape=[],
                                                  name="AUCCallback_value_summary_input")
            self.auc_image_summary = tf.summary.image(name="AUC_plot", tensor=self.auc_image_input)
            self.auc_value_summary = tf.summary.scalar(name="AUC", tensor=self.auc_value_input)
            self.auc_summary_op = tf.summary.merge([self.auc_image_summary, self.auc_value_summary])

        auc_summary_result = self.session.run(self.auc_summary_op, feed_dict={self.auc_image_input: plot_images,
                                                                              self.auc_value_input: auc})

        self.tensorboard.writer.add_summary(auc_summary_result, index)
        self.tensorboard.writer.add_summary(pr_summary_result, index)
