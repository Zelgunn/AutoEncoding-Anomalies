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
        size = [len(inputs)] + [1] * (inputs.ndim - 1)
        if self.gain is not None:
            gain = random_range_value(self.gain, size=size)
            inputs = inputs * gain
            if self.apply_on_outputs:
                outputs = outputs * gain

        if self.bias is not None:
            bias = random_range_value(self.bias, center=0.0, size=size)
            inputs = inputs + bias
            if self.apply_on_outputs:
                outputs = outputs + bias

        if (self.gain is not None) or (self.bias is not None):
            if self.normalization_method == "clip":
                inputs = np.clip(inputs, self.values_range[0], self.values_range[1])
                if self.apply_on_outputs:
                    outputs = np.clip(outputs, self.values_range[0], self.values_range[1])
            elif self.normalization_method == "mod":
                mod_range = self.values_range[1] - self.values_range[0]
                inputs = np.mod(inputs, mod_range) + self.values_range[0]
                if self.apply_on_outputs:
                    outputs = np.mod(outputs, mod_range) + self.values_range[0]

        return inputs, outputs
