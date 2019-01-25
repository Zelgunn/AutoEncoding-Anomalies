from keras.layers import Input, Conv2D, Reshape, Dense, Lambda
import tensorflow as tf
import numpy as np

from models import AutoEncoderBaseModel, KerasModel, metrics_dict


class VAE(AutoEncoderBaseModel):
    def __init__(self):
        super(VAE, self).__init__()
        self.latent_mean = None
        self.latent_log_var = None

    def build_layers(self):
        super(VAE, self).build_layers()
        self.latent_mean = self.embeddings_layer
        self.latent_log_var = Dense(units=self.embeddings_size)
        self.embeddings_layer = Lambda(function=sampling)

    def build_model(self, config_file: str):
        self.load_config(config_file)
        self.build_layers()

    def build_model_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.input_channels]

        with tf.name_scope("model_scale_{0}".format(scale)):
            input_layer = Input(input_shape)
            layer = input_layer

            # region Encoder
            if scale is not (self.depth - 1):
                layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

            with tf.name_scope("encoder"):
                for i in range(scale + 1):
                    layer = self.link_encoder_conv_layer(layer, scale, i)
            # endregion

            # region Embeddings
            with tf.name_scope("embeddings"):
                layer = Reshape([-1])(layer)

                latent_mean = self.latent_mean(layer)
                latent_log_var = self.latent_log_var(layer)
                layer = self.embeddings_layer([latent_mean, latent_log_var])

                layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)
                embeddings_reshape = self.config["embeddings_reshape"]
                embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
                layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)
            # endregion

            # region Decoder
            with tf.name_scope("decoder"):
                for i in range(scale + 1):
                    layer = self.link_decoder_deconv_layer(layer, scale, i)
            # endregion

            output_layer = Conv2D(filters=self.input_channels, kernel_size=1, strides=1, padding="same",
                                  activation=self.output_activation)(layer)

        def vae_loss(y_true, y_pred):
            reconstruction_loss = metrics_dict[self.config["reconstruction_loss"]](y_true, y_pred)
            divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
            return reconstruction_loss + divergence

        model = KerasModel(inputs=input_layer, outputs=output_layer)
        model.compile(self.optimizer, loss=vae_loss, metrics=self.config["metrics"])
        self._models_per_scale[scale] = model
        return model


# region Utils
def sampling(args):
    latent_mean, latent_log_var = args

    shape = tf.shape(latent_mean)
    batch_size = shape[0]
    latent_dim = shape[1]

    epsilon = tf.random_normal(shape=[batch_size, latent_dim], mean=0., stddev=1.0)
    return latent_mean + tf.exp(0.5 * latent_log_var) * epsilon


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
