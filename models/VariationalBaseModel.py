import tensorflow as tf
from keras.layers import Dense, Lambda, Reshape
import keras.backend as K
import numpy as np
import cv2
from abc import ABC
from tqdm import tqdm

from models import AutoEncoderBaseModel, conv_type
from callbacks import CallbackModel, AUCCallback
from datasets import Database, Dataset


class VariationalBaseModel(AutoEncoderBaseModel, ABC):
    def __init__(self):
        super(VariationalBaseModel, self).__init__()
        self.latent_mean_layer = None
        self.latent_log_var_layer = None
        self._latent_mean = None
        self._latent_log_var = None

    def build_layers(self):
        super(VariationalBaseModel, self).build_layers()
        self.latent_mean_layer = self.embeddings_layer
        if self.use_dense_embeddings:
            self.latent_log_var_layer = Dense(units=self.embeddings_size,
                                              kernel_regularizer=self.weight_decay_regularizer,
                                              bias_regularizer=self.weight_decay_regularizer)
        else:
            conv = conv_type["conv_block"][False]
            self.latent_log_var_layer = conv(filters=self.embeddings_size, kernel_size=3, padding="same",
                                             kernel_initializer=self.weights_initializer,
                                             kernel_regularizer=self.weight_decay_regularizer,
                                             bias_regularizer=self.weight_decay_regularizer)

        self.embeddings_layer = Lambda(function=sampling)

    @property
    def latent_mean(self):
        if self._latent_mean is None:
            self.build()
        return self._latent_mean

    @property
    def latent_log_var(self):
        if self._latent_log_var is None:
            self.build()
        return self._latent_log_var

    def compute_embeddings_output_shape(self):
        embeddings_shape = self.compute_embeddings_input_shape()
        if self.use_dense_embeddings:
            embeddings_shape = embeddings_shape[:-1]
            embeddings_shape_prod = np.prod(embeddings_shape)
            assert (self.embeddings_size % embeddings_shape_prod) == 0
            embeddings_filters = self.embeddings_size // embeddings_shape_prod
            embeddings_shape = (*embeddings_shape, embeddings_filters)
        else:
            embeddings_shape = (None, *embeddings_shape)
            embeddings_shape = self.latent_mean_layer.compute_output_shape(embeddings_shape)
            embeddings_shape = embeddings_shape[1:]
        return embeddings_shape

    def train(self,
              database: Database,
              epoch_length: int,
              batch_size: int = 64,
              epochs: int = 1):
        super(VariationalBaseModel, self).train(database, epoch_length, batch_size, epochs)
        self.visualize_vae_interpolation(database.test_dataset)

    def build_anomaly_callbacks(self, database: Database):
        database = self.resize_database(database)
        test_dataset = database.test_dataset
        anomaly_callbacks = super(VariationalBaseModel, self).build_anomaly_callbacks(database)

        samples = test_dataset.sample(batch_size=64, seed=16, sampled_videos_count=16, return_labels=True)
        videos, frame_labels, _ = samples
        videos = test_dataset.divide_batch_io(videos)

        n_predictions_model = self.n_predictions(n=32)

        vae_auc_callback = AUCCallback(n_predictions_model, self.tensorboard,
                                       videos, frame_labels, plot_size=(128, 128), batch_size=1,
                                       name="Variational_AUC", epoch_freq=10)

        anomaly_callbacks.append(vae_auc_callback)
        return anomaly_callbacks

    def n_predictions(self, n):
        with tf.name_scope("n_pred"):
            encoder_input = self.encoder.get_input_at(0)
            true_outputs = self.get_true_outputs_placeholder()
            _, latent_mean, latent_log_var = self.encoder(encoder_input)

            sampling_function = sampling_n(n)
            layer = Lambda(function=sampling_function)([latent_mean, latent_log_var])

            layer = VariationalBaseModel.get_activation(self.embeddings_activation)(layer)
            encoded = Reshape(self.compute_embeddings_output_shape())(layer)

            autoencoded = self.decoder(encoded)

            autoencoded = tf.reshape(autoencoded, [-1, n, *self.output_shape])
            true_outputs_expanded = tf.expand_dims(true_outputs, axis=1)
            error = tf.abs(true_outputs_expanded - autoencoded)
            predictions = tf.reduce_mean(error, axis=[1, -3, -2, -1])

        return CallbackModel(inputs=[encoder_input, true_outputs], outputs=predictions)

    def visualize_vae_interpolation(self, dataset: Dataset):
        encoder_input = self.encoder.get_input_at(0)
        _, latent_mean, latent_log_var = self.encoder(encoder_input)

        decoder_input = self.decoder.get_input_at(0)
        decoder_output = self.decoder.get_output_at(0)

        interpolation_count = 16
        input_video, output_video = dataset.get_batch(batch_size=1, seed=None, apply_preprocess_step=False,
                                                      max_shard_count=1)
        session = K.get_session()

        mean, log_var = session.run([latent_mean, latent_log_var], feed_dict={encoder_input: input_video})
        stddev = np.sqrt(np.exp(log_var))
        z_start = np.random.normal(loc=mean, scale=stddev)
        z_end = np.random.normal(loc=mean, scale=stddev)
        z_shape = [1, *self.compute_decoder_input_shape()]
        z_start = np.reshape(z_start, z_shape)
        z_end = np.reshape(z_end, z_shape)

        decoded = np.empty(shape=[interpolation_count, *self.decoder.output_shape[1:]])
        for i in tqdm(range(interpolation_count), desc="Generating output videos..."):
            progress = i / (interpolation_count - 1.0)
            z = z_start * (1.0 - progress) + z_end * progress

            decoded[i] = session.run(decoder_output, feed_dict={decoder_input: z})

        key = 13
        i = j = 0
        input_video_length = len(input_video[0])
        predicted_video_lenght = len(output_video[0]) - input_video_length
        while key != 27:
            frame = decoded[i][j + input_video_length]
            frame = cv2.resize(frame, (512, 512))
            cv2.imshow("frame", frame)

            base = output_video[0][j + input_video_length]
            base = cv2.resize(base, (512, 512))
            cv2.imshow("base", base)

            key = cv2.waitKey(30)

            j = (j + 1) % predicted_video_lenght
            if key != -1:
                i = (i + 1) % interpolation_count


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
