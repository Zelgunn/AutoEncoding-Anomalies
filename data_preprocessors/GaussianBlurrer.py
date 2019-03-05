import numpy as np
import cv2

from data_preprocessors import DataPreprocessor


class GaussianBlurrer(DataPreprocessor):
    def __init__(self,
                 max_sigma,
                 kernel_size=(5, 5),
                 apply_on_outputs=True):
        super(GaussianBlurrer, self).__init__()
        self.max_sigma = max_sigma
        self.kernel_size = kernel_size
        self.apply_on_outputs = apply_on_outputs

    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        batch_size = inputs.shape[0]
        sigmas = np.random.uniform(low=0.0, high=self.max_sigma, size=batch_size)

        for i in range(batch_size):
            for j in range(inputs.shape[1]):
                cv2.GaussianBlur(inputs[i][j], ksize=self.kernel_size, sigmaX=sigmas[i], dst=inputs[i][j])

            if self.apply_on_outputs:
                for j in range(outputs.shape[1]):
                    cv2.GaussianBlur(outputs[i][j], ksize=self.kernel_size, sigmaX=sigmas[i], dst=outputs[i][j])

        return inputs, outputs
