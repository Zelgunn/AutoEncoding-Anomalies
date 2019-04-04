import tensorflow as tf
import numpy as np
from typing import Dict, List
import os

from datasets import SubsetV2, DatasetConfigV2


class VideoSubsetV2(SubsetV2):
    def __init__(self, records_dict: Dict[str, List[str]], config: DatasetConfigV2):
        super(VideoSubsetV2, self).__init__(config=config)

        self.records_dict = records_dict
        self.videos_filepath = list(self.records_dict.keys())

        self.shards_per_sample = (self.config.output_sequence_length // self.config.shard_size) + 1

    def shard_filepath_generator(self):
        while True:
            video_index = np.random.randint(len(self.videos_filepath))
            video_path = self.videos_filepath[video_index]
            shards = self.records_dict[video_path]
            offset = np.random.randint(len(shards) - self.shards_per_sample + 1)
            yield [os.path.join(video_path, shard) for shard in shards[offset:offset + self.shards_per_sample]]

    def make_tensorflow_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_generator(self.shard_filepath_generator, tf.string)
        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(lambda shard: self.parse_shard(shard))
        dataset = dataset.batch(self.shards_per_sample)
        dataset = dataset.map(lambda shards, shards_length: self.join_shards(shards, shards_length))
        dataset = dataset.batch(batch_size)
        return dataset

    def parse_shard(self, serialized_example):
        features = {"video": tf.VarLenFeature(tf.string)}
        parsed_features = tf.parse_single_example(serialized_example, features)

        encoded_shard = parsed_features["video"].values
        shard_size = tf.shape(encoded_shard)[0]

        images = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_shard[i]), tf.float32),
                           tf.range(shard_size),
                           dtype=tf.float32)

        def images_padded():
            paddings = [[0, self.config.shard_size - shard_size], [0, 0], [0, 0], [0, 0]]
            return tf.pad(images, paddings)

        def images_identity():
            return images

        images = tf.cond(shard_size < self.config.shard_size,
                         images_padded,
                         images_identity)

        return images, shard_size

    def join_shards(self, shards, shards_length):
        total_length = tf.cast(tf.reduce_sum(shards_length), tf.int64)
        offset = tf.random.uniform(shape=(), minval=0, maxval=total_length - self.config.output_sequence_length,
                                   dtype=tf.int64)
        shard_count, shard_size, height, width, channels = tf.unstack(tf.shape(shards))
        shards = tf.reshape(shards, [shard_count * shard_size, height, width, channels])
        shards = shards[offset:offset + self.config.output_sequence_length]
        return shards
