import tensorflow as tf
import numpy as np
import os
from typing import Dict, Tuple

from datasets.loaders import DatasetConfig
from modalities import Modality, ModalityCollection


def get_shard_count(sample_length, shard_size):
    shard_count = 1 + np.ceil((sample_length - 1) / shard_size).astype(np.int)
    return max(1, shard_count)


class SubsetLoader(object):
    def __init__(self,
                 config: DatasetConfig,
                 subset_name: str):
        self.config = config
        self.subset_name = subset_name
        self.subset_files = {folder: sorted(files)
                             for folder, files in config.list_subset_tfrecords(subset_name).items()}
        self.subset_folders = list(self.subset_files.keys())

    def shard_filepath_generator(self):
        while True:
            source_index = np.random.randint(len(self.subset_folders))
            source_folder = self.subset_folders[source_index]
            shards = self.subset_files[source_folder]
            offset = np.random.randint(len(shards) - self.shard_count_per_sample + 1)
            shards_filepath = [os.path.join(source_folder, shard)
                               for shard in shards[offset:offset + self.shard_count_per_sample]]
            for filepath in shards_filepath:
                yield filepath

    def parse_shard(self, serialized_example, output_labels):
        features = self.modalities.get_tfrecord_features()
        if output_labels:
            features["labels"] = tf.VarLenFeature(tf.float32)

        parsed_features = tf.parse_single_example(serialized_example, features)

        modalities_shard_size, features_decoded = {}, {}

        for modality in self.modalities:
            modality_id = modality.tfrecord_id()
            decoded_modality = modality.decode_from_tfrecord_feature(parsed_features)
            decoded_modality, modality_size = self.pad_modality_if_needed(modality, decoded_modality)
            modalities_shard_size[modality_id] = modality_size
            features_decoded[modality_id] = decoded_modality

        if output_labels:
            features_decoded["labels"] = parsed_features["labels"].values

        return features_decoded, modalities_shard_size

    def pad_modality_if_needed(self,
                               modality: Modality,
                               decoded_modality: tf.Tensor
                               ) -> Tuple[tf.Tensor, tf.Tensor]:
        modality_size = tf.shape(decoded_modality)[0]
        modality_max_size = self.config.get_modality_max_shard_size(modality)
        pad_size = modality_max_size - modality_size

        def pad_modality():
            paddings_rank = tf.rank(decoded_modality)
            size_paddings = [[0, pad_size]]
            shape_paddings = tf.zeros(shape=[paddings_rank - 1, 2], dtype=tf.int64)
            paddings = tf.concat([size_paddings, shape_paddings], axis=0,
                                 name=modality.tfrecord_id() + "_paddings")
            return tf.pad(decoded_modality, paddings)

        def identity():
            return decoded_modality

        decoded_modality = tf.cond(pred=pad_size > 0,
                                   true_fn=pad_modality,
                                   false_fn=identity)

        return decoded_modality, modality_size

    def join_shards(self,
                    shards: Dict[str, tf.Tensor],
                    shard_sizes: Dict[str, tf.Tensor]):
        joint_shards = {}
        for modality in self.modalities:
            modality_type = type(modality)
            modality_id = modality.tfrecord_id()
            modality_shards = shards[modality_id]
            modality_shard_sizes = shard_sizes[modality_id]

            total_size = tf.cast(tf.reduce_sum(modality_shard_sizes), tf.int64, name="total_shard_size")
            modality_sample_size = self.modalities[modality_type].io_shape.sample_length
            modality_max_size = self.config.get_modality_max_shard_size(modality)
            size_delta = total_size - modality_sample_size
            offset = tf.random.uniform(shape=(), minval=0, maxval=size_delta, dtype=tf.int64, name="offset")

            modality_shards_shape = [self.shard_count_per_sample, modality_max_size] + [None] * (modality.rank() - 1)
            modality_shards.set_shape(modality_shards_shape)
            # TODO : Shards may have different sizes, even if not including last shard
            # This would happen with a non-integer (floating) frequency (max_size != ceil(size))
            modality_shards = tf.unstack(modality_shards, axis=0)
            modality_shards = tf.concat(modality_shards, axis=0)
            modality_shards = modality_shards[offset:offset + modality_sample_size]

            joint_shards[modality_id] = modality_shards

        if "labels" in shards:
            # TODO : Joint labels, ie: [0, 1], [0, 0.5], [0.5, 1] => [0, 0.5][0.83, 1.0]
            joint_shards["labels"] = tf.reshape(shards["labels"], shape=[-1])

        return joint_shards

    def get_dataset_iterator(self,
                             output_labels: bool,
                             batch_size: int):
        dataset = tf.data.Dataset.from_generator(self.shard_filepath_generator,
                                                 output_types=tf.string,
                                                 output_shapes=())

        dataset = dataset.take(512)

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, output_labels))
        dataset = dataset.batch(self.shard_count_per_sample)
        dataset = dataset.map(self.join_shards)
        dataset = dataset.batch(batch_size)

        return dataset

    @property
    def shard_count_per_sample(self) -> int:
        return self.config.shard_count_per_sample

    @property
    def modalities(self) -> ModalityCollection:
        return self.config.modalities


def main():
    # TODO : Put kitchen sink code here when done
    pass


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
