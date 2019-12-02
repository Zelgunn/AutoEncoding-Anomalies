import tensorflow as tf
from tensorflow import keras
from typing import List, Union

from callbacks import ModalityCallback
from misc_utils.summary_utils import image_summary
from misc_utils.misc_utils import to_list


class ImageCallback(ModalityCallback):
    def __init__(self,
                 inputs: Union[tf.Tensor, List[tf.Tensor]],
                 model: keras.Model,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 outputs: Union[tf.Tensor, List[tf.Tensor]] = None,
                 logged_output_indices=0,
                 name: str = "ImageCallback",
                 max_outputs: int = 4,
                 fps: List[int] = None,
                 **kwargs
                 ):
        super(ImageCallback, self).__init__(inputs=inputs, model=model,
                                            tensorboard=tensorboard, is_train_callback=is_train_callback,
                                            update_freq=update_freq, epoch_freq=epoch_freq,
                                            outputs=outputs, logged_output_indices=logged_output_indices,
                                            name=name, max_outputs=max_outputs, **kwargs)
        if fps is None:
            fps = [25, ]
        elif isinstance(fps, int):
            fps = [fps, ]
        self.fps = fps

        check_image_video_rank(self.true_outputs)
        self.true_outputs = convert_tensors_uint8(self.true_outputs)

    def write_model_summary(self, step: int):
        pred_outputs = self.summary_model(self.inputs)

        pred_outputs = self.extract_logged_modalities(pred_outputs)
        pred_outputs = convert_tensors_uint8(pred_outputs)
        self.samples_summary(data=pred_outputs, step=step, suffix="predicted")

        for i in range(len(self.logged_output_indices)):
            true_sample: tf.Tensor = self.true_outputs[i]
            pred_sample: tf.Tensor = pred_outputs[i]

            if pred_sample.shape.is_compatible_with(true_sample.shape):
                delta = (pred_sample - true_sample) * (tf.cast(pred_sample < true_sample, dtype=tf.uint8) * 254 + 1)
                self.sample_summary(data=delta, step=step, suffix="delta")

    def sample_summary(self, data: tf.Tensor, step: int, suffix: str):
        if use_video_summary(data):
            self.video_summary(data=data, step=step, suffix=suffix)
        else:
            image_summary(name="{}_{}".format(self.name, suffix), data=data, step=step, max_outputs=self.max_outputs)

    def video_summary(self, data: tf.Tensor, step: int, suffix: str):
        for fps in self.fps:
            image_summary(name="{}_{}_{}".format(self.name, fps, suffix), data=data,
                          step=step, max_outputs=self.max_outputs, fps=fps)


# region Utility / wrappers
def check_image_video_rank(data: Union[tf.Tensor, List[tf.Tensor]]):
    if isinstance(data, list) or isinstance(data, tuple):
        for sample in data:
            check_image_video_rank(sample)

    elif data.shape.rank < 4:
        raise ValueError("Incorrect rank for images/video, expected rank >= 4, got {} with rank {}."
                         .format(data.shape, data.shape.rank))


def use_video_summary(data: tf.Tensor) -> bool:
    return data.shape.rank >= 5


def convert_tensors_uint8(tensors: Union[tf.Tensor, List[tf.Tensor]]) -> List[tf.Tensor]:
    tensors = to_list(tensors)
    tensors = [convert_tensor_uint8(tensor) for tensor in tensors]
    return tensors


def convert_tensor_uint8(tensor) -> tf.Tensor:
    tensor: tf.Tensor = tf.convert_to_tensor(tensor)
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized = tf.cast(tensor * tf.constant(255, dtype=tensor.dtype), tf.uint8)
    return normalized
# endregion
