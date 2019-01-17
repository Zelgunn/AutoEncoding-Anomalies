from keras.callbacks import TensorBoard
import numpy as np

from callbacks import TensorBoardPlugin
from models import AutoEncoderBaseModel


class ImageCallback(TensorBoardPlugin):
    def __init__(self,
                 images_summary,
                 tensorboard,
                 feed_dict,
                 update_freq: int or str):
        super(ImageCallback, self).__init__(tensorboard, update_freq)
        self.images_summary = images_summary
        self.feed_dict = feed_dict

    def _write_logs(self, index):
        images_summaries = self.session.run(self.images_summary, feed_dict=self.feed_dict)
        self.writer.add_summary(images_summaries, index)

    @staticmethod
    def from_dataset(images: np.ndarray,
                     autoencoder: AutoEncoderBaseModel,
                     tensorboard: TensorBoard,
                     name: str,
                     update_freq: int or str,
                     scale=None):
        images_indices = [int(i * (images.shape[0] - 1) / (autoencoder.image_summaries_max_outputs - 1))
                          for i in range(autoencoder.image_summaries_max_outputs)]
        images_indices = np.array(images_indices)
        callback_images = images[images_indices]

        if scale is None:
            image_summaries = autoencoder.get_image_summaries(name)
            inputs_placeholder = autoencoder.input
        else:
            image_summaries = autoencoder.get_image_summaries_at_scale(name, scale)
            inputs_placeholder = autoencoder.get_model_at_scale(scale).input

        image_callback = ImageCallback(image_summaries, tensorboard, update_freq=update_freq,
                                       feed_dict={inputs_placeholder: callback_images})
        return image_callback
