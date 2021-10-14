import tensorflow as tf
from typing import Union, List, Tuple

from data_processing import DataProcessor


class VideoProcessor(DataProcessor):
    def __init__(self,
                 target_height: int,
                 target_width: int,
                 to_grayscale: bool,
                 ):
        super(VideoProcessor, self).__init__()

        self.target_height = target_height
        self.target_width = target_width
        self.to_grayscale = to_grayscale

    @tf.function
    def pre_process(self,
                    video: Union[tf.Tensor, List, Tuple]
                    ) -> Union[tf.Tensor, List, Tuple]:
        video = tf.image.resize(video, (self.target_height, self.target_width))

        if self.to_grayscale:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            rgb_weights = tf.reshape(rgb_weights, [1, 1, 1, 3])
            video *= rgb_weights
            video = tf.reduce_sum(video, axis=-1, keepdims=True)

        video = tf.image.per_image_standardization(video)

        return video

    def post_process(self, video: Union[tf.Tensor, List, Tuple]) -> Union[tf.Tensor, List, Tuple]:
        raise NotImplementedError

        # video_min = reduce_min_from(video, start_axis=1, keepdims=True)
        # video_max = reduce_max_from(video, start_axis=1, keepdims=True)
        # video = (video - video_min) / (video_max - video_min)
