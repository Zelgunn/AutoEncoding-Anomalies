import numpy as np

from data_preprocessors import DataPreprocessor


class BrightnessShifter(DataPreprocessor):
    def __init__(self, inputs_gain, inputs_bias, outputs_gain, outputs_bias, values_range=(0.0, 1.0)):
        super(BrightnessShifter, self).__init__()
        self.inputs_gain = inputs_gain
        self.inputs_bias = inputs_bias
        self.outputs_gain = outputs_gain
        self.outputs_bias = outputs_bias
        self.values_range = values_range

    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        inputs_size = [len(inputs)] + [1] * (inputs.ndim - 1)
        outputs_size = [len(outputs)] + [1] * (inputs.ndim - 1)

        if self.inputs_gain > 0.0:
            gain = random_range_value(self.inputs_gain, size=inputs_size)
            inputs = inputs * gain

        if self.inputs_bias > 0.0:
            bias = random_range_value(self.inputs_bias, center=0.0, size=inputs_size)
            inputs = inputs + bias

        if self.inputs_gain > 0.0 or self.inputs_bias > 0.0:
            inputs = np.clip(inputs, self.values_range[0], self.values_range[1])

        if self.outputs_gain > 0.0:
            gain = random_range_value(self.outputs_gain, size=outputs_size)
            outputs = outputs * gain

        if self.outputs_bias > 0.0:
            bias = random_range_value(self.outputs_bias, center=0.0, size=outputs_size)
            outputs = outputs + bias

        if self.outputs_gain > 0.0 or self.outputs_bias > 0.0:
            outputs = np.clip(outputs, self.values_range[0], self.values_range[1])

        return inputs, outputs


def random_range_value(range_or_min_max, center=1.0, size=None):
    if hasattr(range_or_min_max, "__getitem__"):
        min_value, max_value = random_range_value
    else:
        min_value = center - range_or_min_max * 0.5
        max_value = center + range_or_min_max * 0.5

    return np.random.uniform(size=size) * (max_value - min_value) + min_value
