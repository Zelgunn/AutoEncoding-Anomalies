import numpy as np

from data_preprocessors import DataPreprocessor, random_range_value


class BrightnessShifter(DataPreprocessor):
    def __init__(self,
                 gain=None,
                 bias=None,
                 values_range=(0.0, 1.0),
                 normalization_method="clip",
                 apply_on_outputs=True):
        super(BrightnessShifter, self).__init__()
        self.gain = gain
        self.bias = bias
        self.values_range = values_range
        self.normalization_method = normalization_method
        self.apply_on_outputs = apply_on_outputs

    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        if self.apply_on_outputs:
            inputs_len = inputs.shape[1]
            array = np.concatenate([inputs, outputs], axis=1)
            array = self.process_one(array)
            inputs = array[:, :inputs_len]
            outputs = array[:, inputs_len:]
        else:
            inputs = self.process_one(inputs)

        return inputs, outputs

    def process_one(self, array: np.ndarray):
        size = [len(array)] + [1] * (array.ndim - 1)
        if self.gain is not None:
            gain = random_range_value(self.gain, size=size)
            array = array * gain

        if self.bias is not None:
            bias = random_range_value(self.bias, center=0.0, size=size)
            array = array + bias

        if (self.gain is not None) or (self.bias is not None):
            if self.normalization_method == "clip":
                array = np.clip(array, self.values_range[0], self.values_range[1])
            elif self.normalization_method == "mod":
                mod_range = self.values_range[1] - self.values_range[0]
                array = np.mod(array, mod_range) + self.values_range[0]

        return array
