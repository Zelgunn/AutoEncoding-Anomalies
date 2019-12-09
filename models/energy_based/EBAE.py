# EBAE : Energy-based Autoencoder
import tensorflow as tf
from tensorflow_core.python.keras import Model
from typing import Dict, Tuple, Union, List

from models import CustomModel
from models.energy_based import EnergyStateFunction, InputsTensor


class EBAE(CustomModel):
    def __init__(self,
                 autoencoder: CustomModel,
                 energy_margin: float,
                 energy_state_functions: List[EnergyStateFunction],
                 **kwargs
                 ):
        super(EBAE, self).__init__(**kwargs)
        self.autoencoder = autoencoder
        self.energy_margin = energy_margin
        self.low_energy_state_functions = [esf for esf in energy_state_functions if esf.is_low_energy]
        self.high_energy_state_functions = [esf for esf in energy_state_functions if not esf.is_low_energy]

        self.optimizer = self.autoencoder.optimizer

    def __call__(self, *args, **kwargs):
        sum_energies = kwargs.pop("sum_energies") if "sum_energies" in kwargs else False
        energies = super(EBAE, self).__call__(*args, **kwargs)

        if sum_energies and isinstance(energies, (tuple, list)):
            energies = tf.reduce_sum(tf.stack(energies, axis=0), axis=0)

        return energies

    def call(self, inputs, training=None, mask=None) -> Union[tf.Tensor, List[tf.Tensor]]:
        ground_truth, inputs = inputs
        outputs = self.autoencoder(inputs)
        if isinstance(ground_truth, (tuple, list)):
            energies = [self.compute_energy(x, y) for x, y in zip(ground_truth, outputs)]
            return energies
        else:
            energy = self.compute_energy(ground_truth, outputs)
            return energy

    @tf.function
    def compute_energy(self, inputs, outputs):
        reduction_axis = tuple(range(1, inputs.shape.rank))
        return tf.reduce_mean(tf.square(inputs - outputs), axis=reduction_axis)

    @tf.function
    def train_step(self, inputs, *args, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        low_energy_loss = self.train_step_for_energy(inputs, low_energy=True)
        high_energy_loss = self.train_step_for_energy(inputs, low_energy=False)
        total_loss = low_energy_loss + high_energy_loss
        return total_loss, low_energy_loss, high_energy_loss

    def train_step_for_energy(self, inputs, low_energy: bool) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.compute_loss_for_energy(inputs, low_energy)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def compute_loss(self, inputs, *args, **kwargs):
        low_energy_loss = self.compute_loss_for_energy(inputs, low_energy=True)
        high_energy_loss = self.compute_loss_for_energy(inputs, low_energy=False)
        total_loss = low_energy_loss + high_energy_loss
        return total_loss, low_energy_loss, high_energy_loss

    def compute_loss_for_energy(self, inputs, low_energy: bool):
        energy_states = self.get_energy_states(inputs, low_energy=low_energy)

        losses = []
        for state in energy_states:
            energy = self(state, sum_energies=True)
            if not low_energy:
                energy = tf.nn.relu(self.energy_margin - energy)
            losses.append(energy)

        loss = tf.reduce_mean(tf.stack(losses, axis=0), axis=0)
        return loss

    @tf.function
    def forward(self, inputs):
        return self((inputs, inputs))

    def get_energy_states(self, inputs, low_energy: bool) -> List[Tuple[InputsTensor, InputsTensor]]:
        if low_energy:
            energy_states = self.get_low_energy_states(inputs)
        else:
            energy_states = self.get_high_energy_states(inputs)
        return energy_states

    @tf.function
    def get_low_energy_states(self, inputs) -> List[Tuple[InputsTensor, InputsTensor]]:
        states = [func(inputs) for func in self.low_energy_state_functions]
        return states

    @tf.function
    def get_high_energy_states(self, inputs) -> List[Tuple[InputsTensor, InputsTensor]]:
        states = [func(inputs) for func in self.high_energy_state_functions]
        return states

    @property
    def metrics_names(self):
        return ["total_loss", "low_energy_loss", "high_energy_loss"]

    @property
    def models_ids(self) -> Dict[Model, str]:
        return {self.autoencoder: self.autoencoder.name}

    def get_config(self):
        return {
            "autoencoder": self.autoencoder.get_config(),
            "energy_margin": self.energy_margin,
            "low_energy_functions": [str(func) for func in self.low_energy_state_functions],
            "high_energy_functions": [str(func) for func in self.high_energy_state_functions],
        }
