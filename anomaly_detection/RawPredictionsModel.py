from tensorflow.python.keras import Model
from typing import Union, List, Callable

from anomaly_detection import RawPredictionsLayer


class RawPredictionsModel(Model):
    def __init__(self,
                 autoencoder: Callable,
                 output_length: int,
                 metrics: List[Union[str, Callable]] = "mse",
                 **kwargs):
        super(RawPredictionsModel, self).__init__(**kwargs)
        self.autoencoder = autoencoder
        self.output_length = output_length

        if not isinstance(metrics, list):
            metrics = [metrics]

        self.raw_predictions_layers = []

        for metric in metrics:
            layer = RawPredictionsLayer(metric=metric, output_length=output_length)
            self.raw_predictions_layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        inputs, ground_truth = inputs
        decoded = self.autoencoder(inputs)

        predictions = []
        for raw_predictions_layer in self.raw_predictions_layers:
            layer_predictions = raw_predictions_layer([decoded, ground_truth])
            predictions.append(layer_predictions)

        if len(predictions) == 1:
            return predictions[0]
        return predictions

    def compute_output_signature(self, input_signature):
        raise NotImplementedError
