from abc import ABC
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Lambda, Input, Conv2D, Reshape
from typing import List

from models import AutoEncoderBaseModel, AutoEncoderScale, KerasModel
from callbacks import CallbackModel, AUCCallback
from datasets import Database


class VAEScale(AutoEncoderScale):
    def __init__(self,
                 encoder: KerasModel,
                 decoder: KerasModel,
                 autoencoder: KerasModel,
                 latent_mean: KerasModel,
                 latent_log_var: KerasModel,
                 **kwargs):
        super(VAEScale, self).__init__(encoder=encoder,
                                       decoder=decoder,
                                       autoencoder=autoencoder,
                                       **kwargs)
        self.latent_mean = latent_mean
        self.latent_log_var = latent_log_var


class VariationalBaseModel(AutoEncoderBaseModel, ABC):
    # region Initialization
    def __init__(self):
        super(VariationalBaseModel, self).__init__()
        self.latent_mean_layer = None
        self.latent_log_var_layer = None
        self._scales: List[VAEScale] = None

    def load_config(self, config_file: str):
        super(VariationalBaseModel, self).load_config(config_file)

    # endregion

    def build_layers(self):
        super(VariationalBaseModel, self).build_layers()
        self.latent_mean_layer = self.embeddings_layer
        if self.use_dense_embeddings:
            self.latent_log_var_layer = Dense(units=self.embeddings_size,
                                              kernel_regularizer=self.weight_decay_regularizer,
                                              bias_regularizer=self.weight_decay_regularizer)
        else:
            self.latent_log_var_layer = Conv2D(filters=self.embeddings_size, kernel_size=3, padding="same",
                                               kernel_regularizer=self.weight_decay_regularizer,
                                               bias_regularizer=self.weight_decay_regularizer)

        self.embeddings_layer = Lambda(function=sampling)

    # region Model (Getters)
    def get_scale(self, scale: int) -> VAEScale:
        scale: VAEScale = super(VariationalBaseModel, self).get_scale(scale)
        return scale

    def get_latent_distribution_at_scale(self, scale):
        scales = self.get_scale(scale)
        return scales.latent_mean, scales.latent_log_var

    # endregion

    def build_anomaly_callbacks(self,
                                database: Database,
                                scale: int = None):
        scale_shape = self.input_shape_by_scale[scale]
        database = database.resized_to_scale(scale_shape)
        anomaly_callbacks = super(VariationalBaseModel, self).build_anomaly_callbacks(database, scale)

        auc_images, frame_labels = database.test_dataset.sample_with_anomaly_labels(batch_size=64, seed=0,
                                                                                    max_shard_count=8)
        n_predictions_model = self.n_predictions(n=100, scale=scale)

        vae_auc_callback = AUCCallback(n_predictions_model, self.tensorboard,
                                       auc_images, frame_labels, plot_size=(256, 256), batch_size=8,
                                       name="Variational_AUC")

        anomaly_callbacks.append(vae_auc_callback)
        return anomaly_callbacks

    def n_predictions(self, n, scale=None):
        encoder = self.get_encoder_model_at_scale(scale)
        decoder = self.get_decoder_model_at_scale(scale)

        scale_input_shape = self.input_shape_by_scale[scale]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        with tf.name_scope("n_pred"):
            encoder_input = Input(input_shape)
            _, latent_mean, latent_log_var = encoder(encoder_input)

            sampling_function = sampling_n(n)
            layer = Lambda(function=sampling_function)([latent_mean, latent_log_var])

            layer = VariationalBaseModel.get_activation(self.embeddings_activation)(layer)
            embeddings_reshape = self.config["embeddings_reshape"]
            embeddings_filters = self.embeddings_size
            if self.use_dense_embeddings:
                embeddings_filters = embeddings_filters // np.prod(embeddings_reshape)
            encoded = Reshape(embeddings_reshape + [embeddings_filters])(layer)
            autoencoded = decoder(encoded)

            autoencoded = tf.reshape(autoencoded, [-1, n, *input_shape])
            error = tf.abs(tf.expand_dims(encoder_input, axis=1) - autoencoded)
            predictions = tf.reduce_mean(error, axis=[1, 2, 3, 4])

        return CallbackModel(inputs=encoder_input, outputs=predictions)


# region Utils
def sampling(args):
    latent_mean, latent_log_var = args

    shape = tf.shape(latent_mean)
    batch_size = shape[0]
    latent_dim = shape[1]

    epsilon = tf.random_normal(shape=[batch_size, latent_dim], mean=0., stddev=1.0)
    return latent_mean + tf.exp(0.5 * latent_log_var) * epsilon


def sampling_n(n):
    def sampling_function(args):
        latent_mean, latent_log_var = args

        shape = tf.shape(latent_mean)
        batch_size = shape[0]
        latent_dim = shape[1]

        epsilon = tf.random_normal(shape=[batch_size, n, latent_dim], mean=0., stddev=1.0)
        latent_mean = tf.expand_dims(latent_mean, axis=1)
        latent_log_var = tf.expand_dims(latent_log_var, axis=1)
        sample = latent_mean + tf.exp(0.5 * latent_log_var) * epsilon
        sample = tf.reshape(sample, shape=[batch_size * n, latent_dim])
        return sample

    return sampling_function


def kullback_leibler_divergence_mean0_var1(mean, log_variance, use_variance_log=True):
    if use_variance_log:
        divergence = tf.exp(log_variance) + tf.square(mean) - log_variance
    else:
        variance = log_variance
        divergence = variance + tf.square(mean) - tf.log(variance)
    divergence = (tf.reduce_mean(divergence) - 1.0) * 0.5
    return divergence
    # return 0.5 * tf.reduce_mean(-(log_variance + 1) + tf.exp(log_variance) + tf.square(mean))

# endregion
