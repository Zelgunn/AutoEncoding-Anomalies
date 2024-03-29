from tensorflow.python.keras import Model
from typing import Union, List, Callable

from anomaly_detection import IOCompareLayer
from custom_tf_models import AE


class IOCompareModel(Model):
    def __init__(self,
                 autoencoder: Union[Callable, AE],
                 postprocessor: Callable = None,
                 metrics: Union[List[Union[str, Callable]], str, Callable] = "mae",
                 **kwargs):
        super(IOCompareModel, self).__init__(**kwargs)
        self.autoencoder = autoencoder
        self.postprocessor = postprocessor

        if not isinstance(metrics, list):
            metrics = [metrics]

        self.io_compare_layers = []

        for metric in metrics:
            layer = IOCompareLayer(metric=metric)
            self.io_compare_layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        if (len(inputs) % 2) != 0:
            raise ValueError("You must provide an even number of tensors for this model, "
                             "made from [inputs] + [ground_truth].")

        inputs_count = len(inputs) // 2
        inputs, ground_truth = inputs[:inputs_count], inputs[inputs_count:]

        if inputs_count == 1:
            inputs = inputs[0]
            ground_truth = ground_truth[0]

        decoded = self.autoencode(inputs)
        if self.postprocessor is not None:
            decoded = self.postprocessor(decoded)
            ground_truth = self.postprocessor(ground_truth)

        predictions = []
        for io_compare_layer in self.io_compare_layers:
            layer_predictions = io_compare_layer([decoded, ground_truth])
            predictions.append(layer_predictions)

        return predictions

    def autoencode(self, inputs):
        if isinstance(self.autoencoder, AE) or hasattr(self.autoencoder, "autoencode"):
            return self.autoencoder.autoencode(inputs)
        else:
            return self.autoencoder(inputs)

    def compute_output_signature(self, input_signature):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError("IOCompareModel cannot be serialized yet.")
