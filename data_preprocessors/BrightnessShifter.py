import numpy as np

from data_preprocessors import DataPreprocessor


class BrightnessShifter(DataPreprocessor):
    def __init__(self, inputs_gain=None, inputs_bias=None, outputs_gain=None, outputs_bias=None,
                 values_range=(0.0, 1.0), normalization_method="clip"):
        super(BrightnessShifter, self).__init__()
        self.inputs_gain = inputs_gain
        self.inputs_bias = inputs_bias
        self.outputs_gain = outputs_gain
        self.outputs_bias = outputs_bias
        self.values_range = values_range
        self.normalization_method = normalization_method

    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        inputs = self.process_one(inputs, self.inputs_gain, self.inputs_bias)
        outputs = self.process_one(outputs, self.outputs_gain, self.outputs_bias)

        return inputs, outputs

    def process_one(self, array: np.ndarray, gain, bias):
        size = [len(array)] + [1] * (array.ndim - 1)
        if gain is not None:
            gain = random_range_value(gain, size=size)
            array = array * gain

        if bias is not None:
            bias = random_range_value(bias, center=0.0, size=size)
            array = array + bias

        if (gain is not None) or (bias is not None):
            if self.normalization_method == "clip":
                array = np.clip(array, self.values_range[0], self.values_range[1])
            elif self.normalization_method == "mod":
                mod_range = self.values_range[1] - self.values_range[0]
                array = np.mod(array, mod_range) + self.values_range[0]

        return array


def random_range_value(range_or_min_max, center=1.0, size=None):
    if hasattr(range_or_min_max, "__getitem__"):
        min_value, max_value = random_range_value
    else:
        min_value = center - range_or_min_max * 0.5
        max_value = center + range_or_min_max * 0.5

    return np.random.uniform(size=size) * (max_value - min_value) + min_value
