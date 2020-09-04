import tensorflow as tf
from typing import Union, List, Tuple

from data_processing import DataProcessor
from misc_utils.math_utils import standardize_from


class VideoPatchExtractor(DataProcessor):
    def __init__(self, patch_size: int):
        self.patch_size = tf.constant(patch_size, name="patch_size")
        self.original_height = tf.Variable(initial_value=-1, trainable=False, name="original_height")
        self.original_width = tf.Variable(initial_value=-1, trainable=False, name="original_width")

    def pre_process(self, inputs: Union[tf.Tensor, List, Tuple]) -> Union[tf.Tensor, List, Tuple]:
        return inputs

    def batch_process(self,
                      videos: tf.Tensor,
                      labels: tf.Tensor = None,
                      ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        # region Shape variables
        videos_shape = tf.shape(videos)
        batch_size, length, original_height, original_width, channels = tf.unstack(videos_shape)
        patch_count = (original_height // self.patch_size) * (original_width // self.patch_size)
        self.original_height.assign(original_height)
        self.original_width.assign(original_width)
        # endregion

        # videos : [batch_size, length, original_height, original_wid th, channels]
        videos = tf.reshape(videos, [batch_size * length, original_height, original_width, channels])
        patches = tf.image.extract_patches(videos, sizes=self.patch_shape, strides=self.patch_shape,
                                           rates=(1, 1, 1, 1), padding="VALID")
        # patches : [batch_size * length, n_height, n_width, height * width * channels]
        patches = tf.reshape(patches, [batch_size, length, patch_count, self.patch_size, self.patch_size, channels])
        patches = tf.transpose(patches, perm=[0, 2, 1, 3, 4, 5])
        patches = tf.reshape(patches, [batch_size * patch_count, length, self.patch_size, self.patch_size, channels])

        if labels is None:
            return patches
        else:
            return patches, labels

    def post_process(self, videos: Union[tf.Tensor, List, Tuple]) -> Union[tf.Tensor, List, Tuple]:
        # region Shape variables
        n_height = self.original_height // self.patch_size
        n_width = self.original_width // self.patch_size

        input_shape = tf.shape(videos)
        patch_count, length, _, _, channels = tf.unstack(input_shape)
        # endregion

        # videos : [batch_size * n_height * n_width, length, self.height, self.width, channels]
        videos = tf.reshape(videos, [-1, n_height, n_width, length, self.patch_size * self.patch_size * channels])
        videos = tf.transpose(videos, [0, 3, 1, 2, 4])
        videos = tf.reshape(videos, [-1, n_height, n_width, self.patch_size * self.patch_size * channels])
        videos = tf.nn.depth_to_space(videos, self.patch_size, data_format="NHWC")
        videos = tf.reshape(videos, [-1, length, n_height * self.patch_size, n_width * self.patch_size, channels])
        return videos

    @property
    def patch_shape(self) -> Tuple[int, int, int, int]:
        return 1, self.patch_size, self.patch_size, 1
