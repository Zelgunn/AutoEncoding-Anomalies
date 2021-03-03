import tensorflow as tf
from typing import Union, Callable, Optional, Tuple

from data_processing.common import ActivationRange


def make_packets_augmentation(activation_range: Union[ActivationRange, str],
                              gaussian_noise_std=0.0,
                              ) -> Callable[[tf.Tensor], tf.Tensor]:
    def augment_network_packets(network_packets: tf.Tensor) -> tf.Tensor:
        preprocess = make_packets_preprocess(activation_range)
        network_packets = preprocess(network_packets, None)

        if gaussian_noise_std > 0.0:
            video_shape = tf.shape(network_packets)
            network_packets += tf.random.normal(video_shape, stddev=gaussian_noise_std)

            if activation_range == ActivationRange.SIGMOID:
                network_packets = tf.clip_by_value(network_packets, 0.0, 1.0)
            elif activation_range == ActivationRange.TANH:
                network_packets = tf.clip_by_value(network_packets, -1.0, 1.0)

        return network_packets

    return augment_network_packets


def make_packets_preprocess(activation_range: Union[ActivationRange, str]
                            ) -> Callable[[tf.Tensor, Optional[tf.Tensor]],
                                          Tuple[tf.Tensor, Optional[tf.Tensor]]]:
    def preprocess(network_packets: tf.Tensor, labels: tf.Tensor = None):
        if activation_range == ActivationRange.TANH:
            network_packets = tf.multiply(network_packets, 2.0) - 1.0

        if labels is None:
            return network_packets
        else:
            return network_packets, labels

    return preprocess
