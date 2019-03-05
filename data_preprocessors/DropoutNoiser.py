import numpy as np

from data_preprocessors import DataPreprocessor


class DropoutNoiser(DataPreprocessor):
    def __init__(self,
                 inputs_dropout_rate: float,
                 outputs_dropout_rate=0.0):
        self.inputs_dropout_rate = inputs_dropout_rate
        self.outputs_dropout_rate = outputs_dropout_rate
        super(DropoutNoiser, self).__init__()

    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        if self.inputs_dropout_rate > 0.0:
            inputs = add_dropout_noise_to(inputs, self.inputs_dropout_rate)

        if self.outputs_dropout_rate > 0.0:
            outputs = add_dropout_noise_to(outputs, self.outputs_dropout_rate)

        return inputs, outputs


def add_dropout_noise_to(images: np.ndarray, dropout_rate: float):
    original_shape = images.shape
    if dropout_rate > 0.0:
        images = np.reshape(images, [original_shape[0], -1])
        dropout_count = int(np.ceil(dropout_rate * images.shape[-1]))
        for i in range(original_shape[0]):
            dropout_indices = np.random.permutation(np.arange(images.shape[-1]))[:dropout_count]
            images[i][dropout_indices] = 0.0
        images = np.reshape(images, original_shape)
    return images
