import tensorflow as tf
from tensorflow import keras
from typing import List, Union, Callable

from callbacks import ModalityCallback
from misc_utils.summary_utils import network_packet_summary


class NetworkPacketCallback(ModalityCallback):
    def __init__(self,
                 inputs: Union[tf.Tensor, List[tf.Tensor]],
                 model: keras.Model,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 outputs: Union[tf.Tensor, List[tf.Tensor]] = None,
                 compare_to_ground_truth=True,
                 logged_output_indices=0,
                 postprocessor: Callable = None,
                 name: str = "NetworkPacketCallback",
                 max_outputs: int = 4,
                 zoom: int = None,
                 **kwargs
                 ):
        super(NetworkPacketCallback, self).__init__(inputs=inputs, model=model,
                                                    tensorboard=tensorboard, is_train_callback=is_train_callback,
                                                    update_freq=update_freq, epoch_freq=epoch_freq,
                                                    outputs=outputs, logged_output_indices=logged_output_indices,
                                                    name=name, max_outputs=max_outputs, postprocessor=postprocessor,
                                                    **kwargs)

        self.compare_to_ground_truth = compare_to_ground_truth
        self.zoom = zoom

    def write_model_summary(self, step: int):
        pred_outputs = self.summary_model(self.inputs)
        pred_outputs = self.extract_logged_modalities(pred_outputs)
        self.samples_summary(data=pred_outputs, step=step, suffix="predicted")

        if self.compare_to_ground_truth:
            for i in range(len(self.logged_output_indices)):
                true_sample: tf.Tensor = self.true_outputs[i]
                pred_sample: tf.Tensor = pred_outputs[i]

                if pred_sample.shape.is_compatible_with(true_sample.shape):
                    delta = tf.abs(pred_sample - true_sample)
                    self.sample_summary(data=delta, step=step, suffix="delta", cmap=None)

    def sample_summary(self, data: tf.Tensor, step: int, suffix: str, **kwargs):
        name = "{}_{}".format(self.name, suffix)
        normalize = True if "normalize" not in kwargs else kwargs["normalize"]
        cmap = "GnBu" if "cmap" not in kwargs else kwargs["cmap"]

        network_packet_summary(name=name, network_packets=data, step=step, max_outputs=self.max_outputs,
                               zoom=self.zoom, normalize=normalize, cmap=cmap)
