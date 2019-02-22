from abc import ABC
import tensorflow as tf
from keras.layers import Dense, Lambda, Input, Reshape
from typing import List

from models import AutoEncoderBaseModel, AutoEncoderScale, KerasModel, conv_nd
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

    # endregion

    def build_layers(self):
        super(VariationalBaseModel, self).build_layers()
        self.latent_mean_layer = self.embeddings_layer
        if self.use_dense_embeddings:
            self.latent_log_var_layer = Dense(units=self.embeddings_size,
                                              kernel_regularizer=self.weight_decay_regularizer,
                                              bias_regularizer=self.weight_decay_regularizer)
        else:
            conv = conv_nd["conv_block"][False][self.encoder_rank]
            self.latent_log_var_layer = conv(filters=self.embeddings_filters, kernel_size=3, padding="same",
                                             kernel_initializer=self.weights_initializer,
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
        database = self.resize_database(database, scale)
        test_dataset = database.test_dataset
        anomaly_callbacks = super(VariationalBaseModel, self).build_anomaly_callbacks(database, scale)

        samples = test_dataset.sample(batch_size=256, seed=16, max_shard_count=16, return_labels=True)
        auc_images, frame_labels, _ = samples
        auc_images = test_dataset.divide_batch_io(auc_images)

        n_predictions_model = self.n_predictions(n=32, scale=scale)

        vae_auc_callback = AUCCallback(n_predictions_model, self.tensorboard,
                                       auc_images, frame_labels, plot_size=(128, 128), batch_size=4,
                                       name="Variational_AUC", epoch_freq=5)

        anomaly_callbacks.append(vae_auc_callback)
        return anomaly_callbacks

    def n_predictions(self, n, scale):
        encoder = self.get_encoder_model_at_scale(scale)
        decoder = self.get_decoder_model_at_scale(scale)

        scale_output_shape = self.output_shape_by_scale[scale]
        output_shape = scale_output_shape[:-1] + [self.channels_count]

        with tf.name_scope("n_pred"):
            encoder_input = encoder.get_input_at(0)
            true_outputs = self.get_true_outputs_placeholder(scale)
            _, latent_mean, latent_log_var = encoder(encoder_input)

            sampling_function = sampling_n(n)
            layer = Lambda(function=sampling_function)([latent_mean, latent_log_var])

            layer = VariationalBaseModel.get_activation(self.embeddings_activation)(layer)
            encoded = Reshape(self.embeddings_shape)(layer)

            autoencoded = decoder(encoded)

            autoencoded = tf.reshape(autoencoded, [-1, n, *output_shape])
            true_outputs_expanded = tf.expand_dims(true_outputs, axis=1)
            error = tf.abs(true_outputs_expanded - autoencoded)
            predictions = tf.reduce_mean(error, axis=[1, -3, -2, -1])

        return CallbackModel(inputs=[encoder_input, true_outputs], outputs=predictions)


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
