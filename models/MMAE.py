# MMAE : Multi-modal Autoencoder
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from typing import List, Dict, Tuple
from enum import IntEnum

from models import CustomModel, AE


class MMAE(CustomModel):
    def __init__(self,
                 autoencoders: List[AE],
                 learning_rate=1e-3,
                 **kwargs):
        super(MMAE, self).__init__(**kwargs)

        self.autoencoders = autoencoders
        latent_code_sizes = [np.prod(ae.encoder.output_shape[1:]) for ae in autoencoders]
        fusion_model = FusionModel(latent_code_sizes, mode=FusionModelMode.ONE_TO_ONE)

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
        inputs, inputs_shapes, _ = self.split_inputs(inputs, merge_batch_and_steps=True)

        latent_codes = []
        for i in range(self.modality_count):
            latent_code = self.autoencoders[i].encode(inputs[i])
            latent_codes.append(latent_code)

        refined_latent_codes = self.fusion_model([latent_codes, latent_codes])

        outputs = []
        for i in range(self.modality_count):
            output = self.autoencoders[i].decode(refined_latent_codes[i])
            output = tf.reshape(output, inputs_shapes[i])
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

    @tf.function
    def modalities_mse(self, inputs, ground_truths):
        errors = []
        for i in range(self.modality_count):
            error = tf.square(inputs[i] - ground_truths[i])
            errors.append(error)

        errors, _, _ = self.split_inputs(errors, merge_batch_and_steps=False)

        total_error = []
        factors = [1.0, 8.0]
        for i in range(self.modality_count):
            error = errors[i]
            reduction_axis = list(range(2, error.shape.rank))
            error = tf.reduce_mean(error, axis=reduction_axis) * factors[i]
            total_error.append(error)

        total_error = tf.reduce_sum(total_error, axis=0)
        return total_error

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


class FusionModelMode(IntEnum):
    ALL_TO_ONE = 0,
    ONE_TO_ONE = 1


class FusionModel(Model):
    def __init__(self, latent_code_sizes: List[int], mode: FusionModelMode):
        super(FusionModel, self).__init__()

        self.latent_code_sizes = latent_code_sizes
        self.mode = mode
        self.projection_layers: List[Dense] = []

        if self.mode == FusionModelMode.ALL_TO_ONE:
            self.init_all_to_one()
        elif self.mode == FusionModelMode.ONE_TO_ONE:
            self.init_one_to_one()
        else:
            raise ValueError("Unknown mode : {}".format(self.mode))

    def init_all_to_one(self):
        for i, latent_code_size in enumerate(self.latent_code_sizes):
            layer_name = "Project_All_To_{}".format(i)
            self.projection_layers.append(Dense(units=latent_code_size, activation="tanh", name=layer_name))

    def init_one_to_one(self):
        for output_mod_index, latent_code_size in enumerate(self.latent_code_sizes):
            for input_mod_index in range(len(self.latent_code_sizes)):
                if output_mod_index == input_mod_index:
                    continue
                layer_name = "Project_{}_To_{}".format(input_mod_index, output_mod_index)
                self.projection_layers.append(Dense(units=latent_code_size, activation="tanh", name=layer_name))

    def call(self, inputs, training=None, mask=None):
        inputs, fuse_with = inputs

        if self.mode == FusionModelMode.ALL_TO_ONE:
            return self.call_all_to_one(inputs, fuse_with)
        elif self.mode == FusionModelMode.ONE_TO_ONE:
            return self.call_one_to_one(inputs, fuse_with)
        else:
            raise ValueError("Unknown mode : {}".format(self.mode))

    def call_all_to_one(self, inputs, fuse_with=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]

        inputs_latent_codes, fuse_with_latent_codes = self.get_call_flat_latent_codes(inputs, fuse_with)

        fused_latent_codes = []
        for i in range(self.modality_count):
            latent_codes_to_fuse = []
            for j in range(self.modality_count):
                if i == j:
                    code = inputs_latent_codes[i]
                else:
                    code = fuse_with_latent_codes[i]
                latent_codes_to_fuse.append(code)
            fuse_with_latent_codes.append(tf.concat(latent_codes_to_fuse, axis=-1))

        outputs = []
        for i in range(self.modality_count):
            refined_latent_code = self.projection_layers[i](fused_latent_codes[i])
            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[i])
            outputs.append(refined_latent_code)

        return outputs

    def call_one_to_one(self, inputs, fuse_with=None):
        code_shapes = [tf.shape(modality_latent_code) for modality_latent_code in inputs]

        inputs_latent_codes, fuse_with_latent_codes = self.get_call_flat_latent_codes(inputs, fuse_with)

        outputs = []
        i = 0
        for output_mod_index in range(self.modality_count):
            output_mod_latent_code = fuse_with_latent_codes[output_mod_index]
            refined_latent_code = inputs_latent_codes[output_mod_index]
            for input_mod_index in range(self.modality_count):
                if output_mod_index == input_mod_index:
                    continue
                refined_latent_code += self.projection_layers[i](output_mod_latent_code)

                i += 1

            refined_latent_code = tf.reshape(refined_latent_code, code_shapes[output_mod_index])
            outputs.append(refined_latent_code)

        return outputs

    def get_call_flat_latent_codes(self, inputs, fuse_with=None):
        inputs_latent_codes = self.get_flat_latent_codes(inputs)
        if fuse_with is None:
            fuse_with_latent_codes = inputs_latent_codes
        else:
            fuse_with_latent_codes = self.get_flat_latent_codes(fuse_with)
        return inputs_latent_codes, fuse_with_latent_codes

    @tf.function
    def get_flat_latent_codes(self, latent_codes):
        batch_size = tf.shape(latent_codes[0])[0]
        flat_latent_codes = []
        for i in range(self.modality_count):
            latent_code = tf.reshape(latent_codes[i], [batch_size, self.latent_code_sizes[i]])
            flat_latent_codes.append(latent_code)
        return flat_latent_codes

    @property
    def modality_count(self):
        return len(self.projection_layers)
