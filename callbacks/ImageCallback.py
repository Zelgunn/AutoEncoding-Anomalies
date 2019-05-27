import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import def_function
from typing import List, Union

from callbacks import TensorBoardPlugin
from utils.summary_utils import image_summary
from datasets import SubsetLoader


class ImageCallback(TensorBoardPlugin):
    def __init__(self,
                 summary_function: def_function.Function,
                 summary_inputs,
                 tensorboard: keras.callbacks.TensorBoard,
                 is_train_callback: bool,
                 update_freq: int or str,
                 epoch_freq: int = None,
                 is_one_shot=False):
        super(ImageCallback, self).__init__(tensorboard, update_freq, epoch_freq)
        self.summary_function = summary_function
        self.summary_inputs = summary_inputs
        self.writer_name = self.train_run_name if is_train_callback else self.validation_run_name
        self.is_one_shot = is_one_shot

    def _write_logs(self, index):
        if self.is_one_shot and self.summary_function is None:
            return

        with self._get_writer(self.writer_name).as_default():
            if self.summary_function is not None:
                self.summary_function(self.summary_inputs, step=index)

        if self.is_one_shot:
            self.summary_function = None

    @staticmethod
    def convert_tensor_uint8(tensor) -> tf.Tensor:
        tensor: tf.Tensor = tf.convert_to_tensor(tensor)
        normalized = tf.cast(tensor * tf.constant(255, dtype=tensor.dtype), tf.uint8)
        return normalized

    # region Video
    @staticmethod
    def video_summary(name: str,
                      video: tf.Tensor,
                      max_outputs=4,
                      fps=(8, 25),
                      step: int = None):
        video = ImageCallback.convert_tensor_uint8(video)
        for _fps in fps:
            image_summary(name="{}_{}".format(name, _fps),
                          data=video,
                          fps=_fps,
                          step=step,
                          max_outputs=max_outputs,
                          )

    @staticmethod
    def video_autoencoder_summary(name: str,
                                  true_video: tf.Tensor,
                                  pred_video: tf.Tensor,
                                  max_outputs=4,
                                  fps=(8, 25),
                                  step: int = None):
        true_video = ImageCallback.convert_tensor_uint8(true_video)
        pred_video = ImageCallback.convert_tensor_uint8(pred_video)
        delta = (pred_video - true_video) * (tf.cast(pred_video < true_video, dtype=tf.uint8) * 254 + 1)

        for _fps in fps:
            image_summary(name="{}_pred_outputs_{}".format(name, _fps), data=pred_video,
                          step=step, max_outputs=max_outputs, fps=_fps)
            image_summary(name="{}_delta_{}".format(name, _fps), data=delta,
                          step=step, max_outputs=max_outputs, fps=_fps)

    @staticmethod
    def make_video_autoencoder_callbacks(autoencoder: keras.Model,
                                         subset: Union[SubsetLoader],
                                         name: str,
                                         is_train_callback: bool,
                                         tensorboard: keras.callbacks.TensorBoard,
                                         update_freq="epoch",
                                         epoch_freq=1,
                                         ) -> List["ImageCallback"]:
        inputs, outputs = subset.get_batch(batch_size=4, output_labels=False)

        def one_shot_function(data, step):
            return ImageCallback.video_summary(name=name, video=data, step=step)

        def repeated_function(data, step):
            _inputs, _outputs = data
            decoded = autoencoder.predict_on_batch(_inputs)
            return ImageCallback.video_autoencoder_summary(name=name,
                                                           true_video=_outputs,
                                                           pred_video=decoded,
                                                           step=step)

        one_shot_callback = ImageCallback(summary_function=one_shot_function, summary_inputs=inputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=True)

        repeated_callback = ImageCallback(summary_function=repeated_function, summary_inputs=(inputs, outputs),
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=False)

        return [one_shot_callback, repeated_callback]
    # endregion

    # region Images

    @staticmethod
    def image_autoencoder_summary(name: str,
                                  true_image: tf.Tensor,
                                  pred_image: tf.Tensor,
                                  max_outputs=4,
                                  step: int = None):
        true_image = ImageCallback.convert_tensor_uint8(true_image)
        pred_image = ImageCallback.convert_tensor_uint8(pred_image)
        delta = (pred_image - true_image) * (tf.cast(pred_image < true_image, dtype=tf.uint8) * 254 + 1)

        image_summary(name="{}_pred_outputs".format(name), data=pred_image, step=step, max_outputs=max_outputs)
        image_summary(name="{}_delta".format(name), data=delta, step=step, max_outputs=max_outputs)

    @staticmethod
    def make_image_autoencoder_callbacks(autoencoder: keras.Model,
                                         subset: Union[SubsetLoader],
                                         name: str,
                                         is_train_callback: bool,
                                         tensorboard: keras.callbacks.TensorBoard,
                                         update_freq="epoch",
                                         epoch_freq=1,
                                         ) -> List["ImageCallback"]:
        inputs, outputs = subset.get_batch(batch_size=4, output_labels=False)

        def one_shot_function(data, step):
            data = ImageCallback.convert_tensor_uint8(data)
            return image_summary(name=name, data=data, step=step)

        def repeated_function(data, step):
            _inputs, _outputs = data
            decoded = autoencoder.predict_on_batch(_inputs)
            return ImageCallback.image_autoencoder_summary(name=name,
                                                           true_image=_outputs,
                                                           pred_image=decoded,
                                                           step=step)

        one_shot_callback = ImageCallback(summary_function=one_shot_function, summary_inputs=outputs,
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=True)

        repeated_callback = ImageCallback(summary_function=repeated_function, summary_inputs=(inputs, outputs),
                                          tensorboard=tensorboard, is_train_callback=is_train_callback,
                                          update_freq=update_freq, epoch_freq=epoch_freq, is_one_shot=False)

        return [one_shot_callback, repeated_callback]
    # endregion
