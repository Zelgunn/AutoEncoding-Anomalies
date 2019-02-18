from keras.layers import Input, Conv2D, Dense, Reshape
from keras.callbacks import CallbackList
import numpy as np
from typing import List

from models import AutoEncoderBaseModel, AutoEncoderScale, KerasModel, metrics_dict, build_adaptor_layer, LayerBlock
from datasets import Database
from callbacks import AUCCallback, CallbackModel


class GANScale(AutoEncoderScale):
    def __init__(self,
                 encoder: KerasModel,
                 decoder: KerasModel,
                 autoencoder: KerasModel,
                 discriminator: KerasModel,
                 adversarial_generator: KerasModel,
                 **kwargs):
        super(GANScale, self).__init__(encoder=encoder,
                                       decoder=decoder,
                                       autoencoder=autoencoder,
                                       **kwargs)
        self.discriminator = discriminator
        self.adversarial_generator = adversarial_generator


class GAN(AutoEncoderBaseModel):
    def __init__(self):
        super(GAN, self).__init__()
        self.discriminator_layers: List[LayerBlock] = []
        self.discriminator_regression_layer = None
        self._scales: List[GANScale] = []

    # region Model building
    def build_layers(self):
        super(GAN, self).build_layers()

        for layer_info in self.config["discriminator"]:
            layer = self.build_conv_layer_block(layer_info, rank=self.decoder_rank)
            self.discriminator_layers.append(layer)

        self.discriminator_regression_layer = Dense(units=1, activation="sigmoid")

    def build_for_scale(self, scale: int):
        encoder = self.build_encoder_for_scale(scale)
        decoder = self.build_decoder_for_scale(scale)
        discriminator = self.build_discriminator_for_scale(scale)

        scale_input_shape = self.input_shape_by_scale[scale]
        input_shape = scale_input_shape[:-1] + [self.channels_count]

        encoder_input = Input(input_shape)
        autoencoded = decoder(encoder(encoder_input))
        autoencoder = KerasModel(inputs=encoder_input, outputs=autoencoded,
                                 name="AutoEncoder_scale_{0}".format(scale))

        decoder_input = Input(self.embeddings_shape)
        generator_discriminated = discriminator(decoder(decoder_input))
        adversarial_generator = KerasModel(inputs=decoder_input, outputs=generator_discriminated,
                                           name="AdversarialGenerator_scale_{0}".format(scale))

        discriminator_loss_metric = metrics_dict[self.config["losses"]["discriminator"]]

        def discriminator_loss(y_true, y_pred):
            return discriminator_loss_metric(y_true, y_pred) * self.config["loss_weights"]["adversarial"]

        discriminator_metrics = self.config["metrics"]["discriminator"]
        discriminator.compile(self.optimizer, loss=discriminator_loss, metrics=discriminator_metrics)

        reconstruction_metric = metrics_dict[self.config["losses"]["autoencoder"]]

        def autoencoder_loss(y_true, y_pred):
            reconstruction_loss = reconstruction_metric(y_true, y_pred) * self.config["loss_weights"]["reconstruction"]
            return reconstruction_loss

        autoencoder_metrics = self.config["metrics"]["autoencoder"]
        autoencoder.compile(self.optimizer, loss=autoencoder_loss, metrics=autoencoder_metrics)

        discriminator.trainable = False
        adversarial_generator_metrics = self.config["metrics"]["generator"]
        adversarial_generator.compile(self.optimizer, loss=discriminator_loss,
                                      metrics=adversarial_generator_metrics)

        self._scales[scale] = GANScale(encoder, decoder, autoencoder, discriminator, adversarial_generator)

    def build_encoder_for_scale(self, scale: int):
        scale_input_shape = self.input_shape_by_scale[scale]
        scale_channels = scale_input_shape[-1]
        input_shape = scale_input_shape[:-1] + [self.channels_count]

        encoder_name = "Encoder_scale_{0}".format(scale)
        input_layer = Input(input_shape)
        layer = input_layer

        if scale is not (self.depth - 1):
            layer = Conv2D(filters=scale_channels, kernel_size=1, strides=1, padding="same")(layer)

        for i in range(scale + 1):
            layer = self.link_encoder_conv_layer(layer, scale, i)

        if self.use_dense_embeddings:
            layer = Reshape([-1])(layer)
        layer = self.embeddings_layer(layer)
        layer = AutoEncoderBaseModel.get_activation(self.embeddings_activation)(layer)

        outputs = layer
        encoder = KerasModel(inputs=input_layer, outputs=outputs, name=encoder_name)
        return encoder

    def build_decoder_for_scale(self, scale: int):
        decoder_name = "Decoder_scale_{0}".format(scale)
        input_layer = Input(self.embeddings_shape)
        layer = input_layer

        if self.use_dense_embeddings:
            embeddings_reshape = self.config["embeddings_reshape"]
            embeddings_filters = self.embeddings_size // np.prod(embeddings_reshape)
            layer = Reshape(embeddings_reshape + [embeddings_filters])(layer)

        for i in range(scale + 1):
            layer = self.link_decoder_deconv_layer(layer, scale, i)

        layer = build_adaptor_layer(self.channels_count, self.decoder_rank)(layer)
        output_layer = self.get_activation(self.output_activation)(layer)

        decoder = KerasModel(inputs=input_layer, outputs=output_layer, name=decoder_name)
        return decoder

    def build_discriminator_for_scale(self, scale: int):
        scale_output_shape = self.output_shape_by_scale[scale]
        scale_channels = scale_output_shape[-1]
        input_shape = scale_output_shape[:-1] + [self.channels_count]

        discriminator_name = "Discriminator_scale_{0}".format(scale)
        input_layer = Input(input_shape)
        layer = input_layer

        if scale is not (self.depth - 1):
            layer = build_adaptor_layer(scale_channels, rank=self.decoder_rank)(layer)

        for i in range(scale + 1):
            layer = self.link_discriminator_conv_layer(layer, scale, i)

        layer = Reshape([-1])(layer)
        layer = self.discriminator_regression_layer(layer)

        outputs = layer
        discriminator = KerasModel(inputs=input_layer, outputs=outputs, name=discriminator_name)
        return discriminator

    def link_discriminator_conv_layer(self, layer, scale: int, layer_index: int):
        use_dropout = layer_index > 0
        layer_index = layer_index + self.depth - scale - 1
        return self.discriminator_layers[layer_index](layer, use_dropout)

    def get_gan_models_at_scale(self, scale: int = None) -> GANScale:
        if scale is None:
            scale = self.depth - 1
        if self._scales[scale] is None:
            self.build_for_scale(scale)
        return self._scales[scale]

    # endregion

    # region Training
    def train_epoch(self,
                    database: Database,
                    scale: int = None,
                    callbacks: CallbackList = None):
        # region Variables initialization
        epoch_length = len(database.train_dataset)
        scale_models = self.get_gan_models_at_scale(scale)
        autoencoder: KerasModel = scale_models.autoencoder
        decoder: KerasModel = scale_models.decoder
        adversarial_generator: KerasModel = scale_models.adversarial_generator
        discriminator: KerasModel = scale_models.discriminator
        base_model = self.get_autoencoder_model_at_scale(scale)
        discriminator_steps = self.config["discriminator_steps"] if "discriminator_steps" in self.config else 1
        # endregion

        callbacks.on_epoch_begin(self.epochs_seen)
        # discriminator_metrics = [1.0, 0.75]
        for batch_index in range(epoch_length):
            # region Generate batch data (common)
            noisy_x, x = database.train_dataset[0]
            batch_size = x.shape[0]
            x_real = [x]
            z = np.random.normal(size=[discriminator_steps, batch_size] + self.embeddings_shape)
            zeros = np.random.normal(size=[discriminator_steps, batch_size], loc=0.1, scale=0.1)
            zeros = np.clip(zeros, 0.0, 0.3)
            ones = np.random.normal(size=[discriminator_steps, batch_size], loc=0.9, scale=0.1)
            ones = np.clip(ones, 0.7, 1.0)
            # endregion

            # region Generate batch data (discriminator)
            x_generated = []
            for i in range(discriminator_steps):
                if i > 0:
                    x_real += [database.train_dataset.sample(sequence_length=database.output_sequence_length)]
                x_generated += [decoder.predict(x=z[i])]

            x_real = np.array(x_real)
            x_generated = np.array(x_generated)

            instance_noise = np.random.normal(size=[2, *x_real.shape], scale=0.2)

            x_real = np.clip(x_real + instance_noise[0], -1.0, 1.0)
            x_generated = np.clip(x_generated + instance_noise[1], -1.0, 1.0)
            # endregion

            # region Train on Batch
            batch_logs = {"batch": batch_index, "size": batch_size}
            callbacks.on_batch_begin(batch_index, batch_logs)

            autoencoder_metrics = autoencoder.train_on_batch(x=noisy_x, y=x_real[0])
            generator_metrics = adversarial_generator.train_on_batch(x=z[0], y=zeros[0])

            discriminator_metrics = []
            for i in range(discriminator_steps):
                real_data_discriminator_metrics = discriminator.train_on_batch(x=x_real[i], y=zeros[i])
                fake_data_discriminator_metrics = discriminator.train_on_batch(x=x_generated[i], y=ones[i])
                discriminator_metrics += [real_data_discriminator_metrics, fake_data_discriminator_metrics]
            discriminator_metrics = np.mean(discriminator_metrics, axis=0)

            # region Batch logs
            def add_metrics_to_batch_logs(model_name, losses):
                metric_names = ["loss", *self.config["metrics"][model_name]]
                for j in range(len(losses)):
                    batch_logs[model_name + '_' + metric_names[j]] = losses[j]

            add_metrics_to_batch_logs("autoencoder", autoencoder_metrics)
            add_metrics_to_batch_logs("generator", generator_metrics)
            add_metrics_to_batch_logs("discriminator", discriminator_metrics)
            # endregion

            callbacks.on_batch_end(batch_index, batch_logs)
            # endregion

        self.on_epoch_end(base_model, database, callbacks)

    # endregion

    # region Callbacks
    def build_anomaly_callbacks(self,
                                database: Database,
                                scale: int = None):
        database = self.resize_database(database, scale)
        test_dataset = database.test_dataset
        anomaly_callbacks = super(GAN, self).build_anomaly_callbacks(database, scale)

        discriminator: KerasModel = self.get_gan_models_at_scale(scale).discriminator
        discriminator_prediction = discriminator.get_output_at(0)
        discriminator_inputs_placeholder = discriminator.get_input_at(0)

        samples = test_dataset.sample(batch_size=512, seed=16, max_shard_count=8, return_labels=True)
        auc_images, frame_labels, _ = samples

        disc_auc_predictions_model = CallbackModel(discriminator_inputs_placeholder, discriminator_prediction)

        disc_auc_callback = AUCCallback(disc_auc_predictions_model, self.tensorboard,
                                        auc_images, frame_labels, plot_size=(256, 256), batch_size=128,
                                        name="Discriminator_AUC")

        anomaly_callbacks.append(disc_auc_callback)
        return anomaly_callbacks

    def callback_metrics(self, model: KerasModel):
        def model_metric_names(model_name):
            metric_names = ["loss", *self.config["metrics"][model_name]]
            return [model_name + "_" + metric_name for metric_name in metric_names]

        return model_metric_names("autoencoder") + model_metric_names("generator") + model_metric_names("discriminator")
    # endregion
