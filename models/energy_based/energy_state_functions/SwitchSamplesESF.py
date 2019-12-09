import tensorflow as tf

from models.energy_based import ApplyOnRandomInput


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