import tensorflow as tf
from tensorflow.python.keras import backend, Model
from typing import Dict

from models import CustomModel, AE


class AND(CustomModel):
    def __init__(self,
                 autoencoder: AE,
                 autoregressive_model: Model,
                 beta=1.0,
                 learning_rate=1e-3,
                 **kwargs
                 ):
        super(AND, self).__init__(**kwargs)

        self.autoencoder = autoencoder
        self.autoregressive_model = autoregressive_model
        self.beta = beta
        self.learning_rate = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs, training=None, mask=None):
        return self.autoencoder(inputs)

    @tf.function
    def train_step(self, inputs, *args, **kwargs):
        with tf.GradientTape() as tape:
            reconstruction_loss, z_loss = self.compute_loss(inputs)
            loss = reconstruction_loss + self.beta * z_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, reconstruction_loss, z_loss

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        z = self.autoencoder.encode(inputs)
        reconstructed = self.autoencoder.decode(z)
        z_predicted_distribution = self.autoregressive_model(tf.expand_dims(z, axis=-1))

        reconstruction_loss = self.compute_reconstruction_loss(inputs, reconstructed)
        z_loss = self.compute_z_loss(z, z_predicted_distribution)

        return reconstruction_loss, z_loss

    @tf.function
    def compute_reconstruction_loss(self, inputs, reconstructed):
        error = tf.square(inputs - reconstructed)
        reduction_axis = list(range(1, error.shape.rank))
        error = tf.reduce_sum(error, axis=reduction_axis)
        return tf.reduce_mean(error)

    @tf.function
    def compute_z_loss(self, z, z_predicted_distribution):
        z_predicted_distribution_shape = tf.shape(z_predicted_distribution)
        batch_size = z_predicted_distribution_shape[0]
        n_bins = z_predicted_distribution_shape[-1]

        z_predicted_distribution = tf.nn.softmax(z_predicted_distribution, axis=1)
        z_predicted_distribution = tf.reshape(z_predicted_distribution, [batch_size, -1, n_bins])
        epsilon = backend.epsilon()
        z_predicted_distribution = tf.clip_by_value(z_predicted_distribution, epsilon, 1.0 - epsilon)
        z_predicted_distribution = tf.math.log(z_predicted_distribution)

        z = tf.reshape(z, [batch_size, -1, 1])
        n_bins = tf.cast(n_bins, tf.float32)
        z = tf.clip_by_value(z * n_bins, 0.0, n_bins - 1.0)
        z = tf.cast(z, tf.int32)

        selected_bins = tf.gather(z_predicted_distribution, indices=z, batch_dims=-1)

        z_loss = tf.reduce_sum(selected_bins, axis=-2)
        z_loss = - tf.reduce_mean(z_loss)

        return z_loss

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **self.autoencoder.models_ids,
            self.autoregressive_model: "autoregressive_model"
        }

    @property
    def metrics_names(self):
        return ["loss", "reconstruction_loss", "z_loss"]


def main():
    from models.autoregressive.SAAM import SAAM
    from models import AE
    from z_kitchen_sink.video_cnn_transformer import make_encoder, make_decoder
    from tensorflow.python.keras.layers import Input

    input_layer = Input(shape=[8, 32, 32, 1])
    common_cnn_params = {"padding": "same", "activation": "relu", "kernel_initializer": "glorot_uniform",
                         "use_bias": True}
    encoder = make_encoder(input_layer, code_size=64, common_cnn_params=common_cnn_params)

    decoder_input_layer = Input(shape=[2, 4, 4, 64], name="DecoderInputLayer")
    decoder = make_decoder(decoder_input_layer, 1, common_cnn_params)

    ae = AE(encoder, decoder)

    saam = SAAM(layer_count=4, head_count=4, head_size=32, intermediate_size=4, output_size=100)

    and_model = AND(ae, saam)

    x = tf.nn.sigmoid(tf.random.normal([4, 8, 32, 32, 1]))
    loss, reconstruction_loss, z_loss = and_model.train_step(x)
    print(loss, reconstruction_loss, z_loss)
    exit()


if __name__ is "__main__":
    main()
