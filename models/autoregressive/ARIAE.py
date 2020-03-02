# ARIAE : Autoregressive Interpolating Autoencoder
import tensorflow as tf
from tensorflow_core.python.keras import Model
from typing import Dict

from transformers import Transformer

from models import CustomModel, IAE


class ARIAE(CustomModel):
    def __init__(self,
                 iae: IAE,
                 transformer: Transformer,
                 **kwargs
                 ):
        super(ARIAE, self).__init__(**kwargs)

        self.iae = iae
        self.transformer = transformer

        self.iae.trainable = False

    def train_step(self, inputs, *args, **kwargs):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def compute_loss(self, inputs, *args, **kwargs) -> tf.Tensor:
        inputs, inputs_shape, split_shape = self.iae.split_inputs(inputs, merge_batch_and_steps=True)
        encoded = self.iae.encode(inputs)

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {
            **self.iae.models_ids,
            self.transformer: "transformer"
        }

    def get_config(self):
        pass
