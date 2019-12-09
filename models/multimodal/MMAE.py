# MMAE : Multi-modal Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
import numpy as np
from typing import List, Dict, Tuple

from models import CustomModel, AE
from models.multimodal import DenseFusionModel, DenseFusionModelMode


class MMAE(CustomModel):
    def __init__(self,
                 autoencoders: List[AE],
                 learning_rate=1e-3,
                 **kwargs):
        super(MMAE, self).__init__(**kwargs)

        self.autoencoders = autoencoders
        latent_code_sizes = [np.prod(ae.encoder.output_shape[1:]) for ae in autoencoders]
        fusion_model = DenseFusionModel(latent_code_sizes, mode=DenseFusionModelMode.ONE_TO_ONE)

        fusion_base_input_layers = [Input(shape=[code_size], name="FusionInputBase_{}".format(i))
                                    for i, code_size in enumerate(latent_code_sizes)]

        fuse_with_input_layers = [Input(shape=[code_size], name="FusionInputFuseWith_{}".format(i))
                                  for i, code_size in enumerate(latent_code_sizes)]

        fusion_input_layers = [fusion_base_input_layers, fuse_with_input_layers]

        fusion_output_layers = fusion_model(fusion_input_layers)

        self.fusion_model = Model(inputs=fusion_input_layers, outputs=fusion_output_layers, name="FusionModelProxy")

        self.optimizer = None
        self.set_optimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def call(self, inputs, training=None, mask=None):
        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.autoencoders[i].encode(inputs[i])
            latent_codes.append(latent_code)

        refined_latent_codes = self.fusion_model([latent_codes, latent_codes])

        outputs = []
        for i in range(self.modality_count):
            output = self.autoencoders[i].decode(refined_latent_codes[i])
            output = tf.reshape(output, tf.shape(inputs[i]))
            outputs.append(output)

        return outputs

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self.compute_loss(inputs)
            total_loss = tf.reduce_sum(losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return (total_loss, *losses)

    @tf.function
    def compute_loss(self,
                     inputs
                     ) -> Tuple[tf.Tensor]:
        outputs = self(inputs)

        losses = []
        for i in range(self.modality_count):
            modality_loss: tf.Tensor = tf.reduce_mean(tf.square(inputs[i] - outputs[i]))
            losses.append(modality_loss)

        return tuple(losses)

    @property
    def modality_count(self):
        return len(self.autoencoders)

    @property
    def metrics_names(self):
        return ["reconstruction"] + [ae.name for ae in self.autoencoders]

    def get_config(self):
        config = {ae.name: ae.get_config() for ae in self.autoencoders}
        return config

    @property
    def models_ids(self) -> Dict[Model, str]:
        ids = {ae: ae.name for ae in self.autoencoders}
        ids[self.fusion_model] = self.fusion_model.name
        return ids

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for ae in self.autoencoders:
            ae.set_optimizer(optimizer)





