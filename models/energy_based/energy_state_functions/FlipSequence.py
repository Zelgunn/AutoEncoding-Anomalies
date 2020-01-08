import tensorflow as tf

from models.energy_based import ApplyOnRandomInput


class FlipSequence(ApplyOnRandomInput):
    def __init__(self):
        super(FlipSequence, self).__init__(is_low_energy=False,
                                           ground_truth_from_inputs=True)

    def apply_on_one(self, input_tensor):
        return tf.reverse(input_tensor, axis=(1,))

    def apply_on_others(self, input_tensor):
        return input_tensor
