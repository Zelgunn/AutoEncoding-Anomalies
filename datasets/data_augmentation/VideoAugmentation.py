import tensorflow as tf
from datasets.data_augmentation import DataAugmentation


class VideoAugmentation(DataAugmentation):
    def __init__(self,
                 gain=None,
                 bias=None,
                 values_range=(0.0, 1.0),
                 apply_on_outputs=True):
        super(VideoAugmentation, self).__init__()
        self.gain = gain
        self.bias = bias
        self.values_range = values_range
        self.apply_on_outputs = apply_on_outputs

    def process(self, inputs: tf.data.Dataset, outputs: tf.data.Dataset):
        size = [len(inputs)] + [1] * (inputs.ndim - 1)
        if self.gain is not None:
            gain = random_range_value(self.gain, size=size)
            inputs *= gain
            if self.apply_on_outputs:
                outputs *= gain

        if self.bias is not None:
            bias = random_range_value(self.bias, center=0.0, size=size)
            inputs += bias
            if self.apply_on_outputs:
                outputs += bias

        if (self.gain is not None) or (self.bias is not None):
            inputs.clip(self.values_range[0], self.values_range[1], out=inputs)
            if self.apply_on_outputs:
                outputs.clip(self.values_range[0], self.values_range[1], out=outputs)

        return inputs, outputs
