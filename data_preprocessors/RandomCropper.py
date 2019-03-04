import numpy as np
import cv2

from data_preprocessors import DataPreprocessor


class RandomCropper(DataPreprocessor):
    def __init__(self,
                 width_range,
                 height_range=None,
                 keep_ratio=True):
        super(RandomCropper, self).__init__()
        self.width_range = width_range
        self.height_range = height_range
        self.keep_ratio = keep_ratio

    def process(self, inputs: np.ndarray, outputs: np.ndarray):
        batch_size, inputs_length, inputs_width, inputs_height, channels = inputs.shape
        outputs_length = outputs.shape[1]

        width_ratio = np.random.uniform(size=batch_size, low=1.0 - self.width_range, high=1.0)
        width_offsets = np.random.uniform(size=batch_size, low=0.0, high=1.0)
        height_offsets = np.random.uniform(size=batch_size, low=0.0, high=1.0)
        if self.keep_ratio:
            height_ratio = width_ratio
        else:
            height_ratio = np.random.uniform(size=batch_size, low=1.0 - self.height_range, high=1.0)

        widths = width_ratio * inputs_width
        heights = height_ratio * inputs_height
        widths = np.round(widths).astype(np.int32)
        heights = np.round(heights).astype(np.int32)

        width_offsets = (inputs_width - widths) * width_offsets
        height_offsets = (inputs_height - heights) * height_offsets
        width_offsets = np.round(width_offsets).astype(np.int32)
        height_offsets = np.round(height_offsets).astype(np.int32)

        for i in range(batch_size):
            y_start, y_end = height_offsets[i], height_offsets[i] + heights[i]
            x_start, x_end = width_offsets[i], width_offsets[i] + widths[i]

            for j in range(inputs_length):
                inputs[i][j] = crop_and_resize_frame(inputs[i][j], x_start, x_end, y_start, y_end)

            for j in range(outputs_length):
                outputs[i][j] = crop_and_resize_frame(outputs[i][j], x_start, x_end, y_start, y_end)

        return outputs, inputs


def crop_and_resize_frame(frame: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int):
    dsize = (frame.shape[1], frame.shape[0])

    cropped_frame = frame[y_start: y_end, x_start: x_end]
    resized_frame = cv2.resize(cropped_frame, dsize)

    if (frame.ndim == 3) and (frame.shape[-1] == 1):
        resized_frame = np.expand_dims(resized_frame, axis=-1)

    return resized_frame
