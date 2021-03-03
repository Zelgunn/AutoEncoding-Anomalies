import tensorflow as tf
from tensorflow import keras
from typing import List, Union, Callable

from callbacks import ModalityCallback
from misc_utils.summary_utils import image_summary, convert_tensors_uint8, check_image_video_rank, use_video_summary


class ImageCallback(ModalityCallback):
    def __init__(self,
                 inputs: Union[tf.Tensor, List[tf.Tensor]],
                 model: keras.Model,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 outputs: Union[tf.Tensor, List[tf.Tensor]] = None,
                 compare_to_ground_truth=True,
                 logged_output_indices=0,
                 postprocessor: Callable = None,
                 name: str = "ImageCallback",
                 max_outputs: int = 4,
                 video_sample_rate: List[int] = None,
                 **kwargs
                 ):
        super(ImageCallback, self).__init__(inputs=inputs, model=model,
                                            tensorboard=tensorboard, is_train_callback=is_train_callback,
                                            update_freq=update_freq, epoch_freq=epoch_freq,
                                            outputs=outputs, logged_output_indices=logged_output_indices,
                                            name=name, max_outputs=max_outputs, postprocessor=postprocessor,
                                            **kwargs)
        if video_sample_rate is None:
            video_sample_rate = [25, ]
        elif isinstance(video_sample_rate, int):
            video_sample_rate = [video_sample_rate, ]
        self.video_sample_rate = video_sample_rate
        self.compare_to_ground_truth = compare_to_ground_truth

        check_image_video_rank(self.true_outputs)
        self.true_outputs = convert_tensors_uint8(self.true_outputs)

    def write_model_summary(self, step: int):
        pred_outputs = self.summary_model(self.inputs)
        if self.postprocessor is not None:
            pred_outputs = self.postprocessor(pred_outputs)

        pred_outputs = self.extract_logged_modalities(pred_outputs)
        pred_outputs = convert_tensors_uint8(pred_outputs)
        self.samples_summary(data=pred_outputs, step=step, suffix="predicted")

        if self.compare_to_ground_truth:
            for i in range(len(self.logged_output_indices)):
                true_sample: tf.Tensor = self.true_outputs[i]
                pred_sample: tf.Tensor = pred_outputs[i]

                if pred_sample.shape.is_compatible_with(true_sample.shape):
                    delta = self.images_uint8_delta(true_sample, pred_sample)
                    self.sample_summary(data=delta, step=step, suffix="delta")

    def sample_summary(self, data: tf.Tensor, step: int, suffix: str, **kwargs):
        if use_video_summary(data):
            self.video_summary(data=data, step=step, suffix=suffix)
        else:
            image_summary(name="{}_{}".format(self.name, suffix), data=data, step=step, max_outputs=self.max_outputs)

    def video_summary(self, data: tf.Tensor, step: int, suffix: str):
        for video_sample_rate in self.video_sample_rate:
            image_summary(name="{}_{}_{}".format(self.name, video_sample_rate, suffix), data=data,
                          step=step, max_outputs=self.max_outputs, fps=video_sample_rate)

    @staticmethod
    def images_uint8_delta(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        delta = (y - x) * (tf.cast(y < x, dtype=tf.uint8) * 254 + 1)
        return delta
