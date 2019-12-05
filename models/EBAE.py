# EBAE : Energy-based Autoencoder
import tensorflow as tf
from tensorflow_core.python.keras import Model
from abc import abstractmethod
from typing import Dict, Tuple, Union, List

from models import CustomModel

InputsTensor = Union[tf.Tensor, List[tf.Tensor], List[List[tf.Tensor]]]


class EBAE(CustomModel):
    def __init__(self,
                 autoencoder: CustomModel,
                 energy_margin: float,
                 energy_state_functions: List["EnergyStateFunction"],
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


class EnergyStateFunction(object):
    def __init__(self,
                 is_low_energy: bool,
                 ground_truth_from_inputs: bool,
                 ):
        self.is_low_energy = is_low_energy
        self.ground_truth_from_inputs = ground_truth_from_inputs

    def __call__(self,
                 inputs: InputsTensor
                 ) -> Tuple[InputsTensor, InputsTensor]:
        state = self.call(inputs)
        if self.ground_truth_from_inputs:
            inputs = ground_truth = state
        else:
            inputs, ground_truth = state
        return inputs, ground_truth

    @abstractmethod
    def call(self,
             inputs: InputsTensor
             ) -> Union[Tuple[InputsTensor, InputsTensor], InputsTensor]:
        raise NotImplementedError


class IdentityESF(EnergyStateFunction):
    def __init__(self):
        super(IdentityESF, self).__init__(is_low_energy=True,
                                          ground_truth_from_inputs=True)

    def call(self, inputs):
        return inputs


class TakeStepESF(EnergyStateFunction):
    def __init__(self,
                 step_count: int,
                 axis=1,
                 ):
        super(TakeStepESF, self).__init__(is_low_energy=True,
                                          ground_truth_from_inputs=True)
        self.step_count = step_count
        self.axis = axis

    def call(self, inputs):
        multiple_inputs = isinstance(inputs, (tuple, list))

        if multiple_inputs:
            state = self.take_inputs_steps(inputs)
        else:
            state = self.take_step(inputs)

        return state

    @tf.function
    def take_inputs_steps(self, inputs):
        outputs = []
        for x in inputs:
            x = self.take_step(x)
            outputs.append(x)
        return outputs

    @tf.function
    def take_step(self, inputs: tf.Tensor):
        step_size = tf.shape(inputs)[self.axis] // self.step_count
        return inputs[:, :step_size]


class ApplyOnRandomInput(EnergyStateFunction):
    def call(self, inputs):
        outputs = []
        index = tf.random.uniform(shape=[], minval=0, maxval=len(inputs), dtype=tf.int32)
        for i in range(len(inputs)):
            x = inputs[i]
            if i == index:
                output = self.apply_on_one(x)
            else:
                output = self.apply_on_others(x)
            outputs.append(output)
        return outputs

    @abstractmethod
    def apply_on_one(self, input_tensor):
        raise NotImplementedError

    @abstractmethod
    def apply_on_others(self, input_tensor):
        raise NotImplementedError


class OffsetSequences(TakeStepESF, ApplyOnRandomInput):
    def __init__(self, step_count: int):
        super(OffsetSequences, self).__init__(step_count=step_count)
        self.is_low_energy = False

    def apply_on_one(self, input_tensor):
        return self.offset_sequence(input_tensor)

    def apply_on_others(self, input_tensor):
        return self.take_step(input_tensor)

    @tf.function
    def offset_sequence(self, sequence: tf.Tensor):
        sequence_rank = sequence.shape.rank
        if self.axis < 0:
            axis = sequence_rank + self.axis
        else:
            axis = self.axis

        length = tf.shape(sequence)[axis]
        step_size = length // self.step_count

        min_offset = step_size // 2
        max_offset = length - step_size
        offset = tf.random.uniform(shape=[], minval=min_offset, maxval=max_offset + 1, dtype=tf.int32)

        begin = tf.pad([offset], paddings=[[axis, sequence_rank - axis - 1]], constant_values=0)
        end = tf.pad([step_size], paddings=[[axis, sequence_rank - axis - 1]], constant_values=-1)
        sequence = tf.slice(sequence, begin=begin, size=end)
        return sequence


class SwitchSamplesESF(ApplyOnRandomInput):
    def __init__(self, axis=0):
        super(SwitchSamplesESF, self).__init__(is_low_energy=False,
                                               ground_truth_from_inputs=True)
        self.axis = axis

    def apply_on_one(self, input_tensor):
        return self.switch_samples(input_tensor)

    def apply_on_others(self, input_tensor):
        return input_tensor

    def switch_samples(self, input_tensor):
        input_tensor = tf.reverse(input_tensor, axis=(self.axis,))
        return input_tensor


class CombineESF(EnergyStateFunction):
    def __init__(self,
                 energy_state_functions: List[EnergyStateFunction],
                 is_low_energy: bool = None,
                 ground_truth_from_inputs: bool = None,
                 ):
        self.energy_state_functions = energy_state_functions

        if is_low_energy is None:
            is_low_energy = all([esf.is_low_energy for esf in energy_state_functions])

        self.raise_if_ground_truth_from_inputs_expect_for_last(energy_state_functions)

        if ground_truth_from_inputs is None:
            ground_truth_from_inputs = energy_state_functions[-1].ground_truth_from_inputs

        super(CombineESF, self).__init__(is_low_energy=is_low_energy,
                                         ground_truth_from_inputs=ground_truth_from_inputs)

    def call(self,
             inputs: InputsTensor
             ) -> Union[Tuple[InputsTensor, InputsTensor], InputsTensor]:
        for esf in self.energy_state_functions:
            inputs = esf.call(inputs)
        return inputs

    @staticmethod
    def raise_if_ground_truth_from_inputs_expect_for_last(energy_state_functions: List[EnergyStateFunction]):
        for esf in energy_state_functions[:-1]:
            if not esf.ground_truth_from_inputs:
                raise ValueError("At the moment, only the last provided EnergyStateFunction can "
                                 "output different ground_truth from inputs. Got `false` for {}.".format(esf))
