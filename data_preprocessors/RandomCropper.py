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
        batch_size, inputs_length, inputs_height, inputs_width, channels = inputs.shape
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

        cropped_inputs = np.empty(shape=[batch_size, inputs_length, 128, 128, channels])
        cropped_outputs = np.empty(shape=[batch_size, outputs_length, 128, 128, channels])

        for i in range(batch_size):
            x_start, x_end = height_offsets[i], height_offsets[i] + heights[i]
            y_start, y_end = width_offsets[i], width_offsets[i] + widths[i]

            for j in range(inputs_length):
                frame = crop_and_resize_frame(inputs[i][j], x_start, x_end, y_start, y_end, dsize=(128, 128))
                cropped_inputs[i][j] = np.expand_dims(frame, axis=-1)

            for j in range(outputs_length):
                frame = crop_and_resize_frame(outputs[i][j], x_start, x_end, y_start, y_end, dsize=(128, 128))
                cropped_outputs[i][j] = np.expand_dims(frame, axis=-1)

        return cropped_inputs, cropped_outputs


def crop_and_resize_frame(frame: np.ndarray, x_start: int, x_end: int, y_start: int, y_end: int, dst=None, dsize=None):
    if dsize is None:
        dsize = (frame.shape[1], frame.shape[0])

    cropped_frame = frame[x_start: x_end, y_start: y_end]
    resized_frame = cv2.resize(cropped_frame, dsize, dst=dst)

    return resized_frame
