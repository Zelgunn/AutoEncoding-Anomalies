import tensorflow as tf
from tensorflow.python.keras.backend import get_session
import numpy as np
import os
from typing import Dict, Tuple, Optional

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

        for modality in self.config.modalities:
            shard_size = modality.frequency * self.config.shard_duration
            max_shard_size = int(np.ceil(shard_size))
            initial_shard_size = int(np.floor(shard_size))
            if max_shard_size != initial_shard_size:
                raise NotImplementedError("SubsetLoader doesn't support this case yet.")

        self._train_tf_dataset: Optional[tf.data.Dataset] = None
        self._test_tf_dataset: Optional[tf.data.Dataset] = None

        self._train_iterators: Dict[int, tf.data.Iterator] = {}
        self._test_iterators: Dict[int, tf.data.Iterator] = {}

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

    # region Loading methods (parsing, decoding, fusing, normalization, splitting, ...)
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
            labels = parsed_features["labels"].values
            labels = self.pad_labels_if_needed(labels)
            features_decoded["labels"] = labels

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

    def pad_labels_if_needed(self, labels: tf.Tensor):
        labels_size = tf.shape(labels)[0]
        max_labels_size = self.config.max_labels_size
        pad_size = max_labels_size - labels_size

        def pad_labels():
            paddings = [[pad_size, 0]]
            return tf.pad(labels, paddings)

        def identity():
            return labels

        labels = tf.cond(pred=pad_size > 0,
                         true_fn=pad_labels,
                         false_fn=identity)

        return labels

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
            modality_shards = tf.unstack(modality_shards, axis=0)
            modality_shards = tf.concat(modality_shards, axis=0)
            modality_shards = modality_shards[offset:offset + modality_sample_size]

            joint_shards[modality_id] = modality_shards

        if "labels" in shards:
            labels = shards["labels"]
            shard_labels_offset = tf.range(self.shard_count_per_sample, dtype=tf.float32)
            shard_labels_offset = tf.reshape(shard_labels_offset, [self.shard_count_per_sample, 1])
            labels = (labels + shard_labels_offset) / self.shard_count_per_sample
            labels = tf.reshape(labels, shape=[-1, 2])
            joint_shards["labels"] = labels

        return joint_shards

    # TODO : Normalize according to config (model output)
    def normalize_batch(self, modalities: Dict[str, tf.Tensor]):
        for modality_id in modalities:
            modality_value = modalities[modality_id]
            modality_min, modality_max = self.config.modalities_ranges[modality_id]
            modalities[modality_id] = (modality_value - modality_min) / (modality_max - modality_min)
        return modalities

    # TODO : What about labels ?
    def split_batch_io(self, modalities: Dict[str, tf.Tensor]):
        inputs, outputs = [], []
        for modality_id, modality_value in modalities.items():
            modality_type = ModalityCollection.modality_id_to_class(modality_id)
            input_length = self.modalities[modality_type].io_shape.input_length
            output_length = self.modalities[modality_type].io_shape.output_length
            inputs.append(modality_value[:input_length])
            outputs.append(modality_value[-output_length:])
        return tuple(inputs), tuple(outputs)

    # endregion

    def make_tf_dataset(self, output_labels: bool):
        dataset = tf.data.Dataset.from_generator(self.shard_filepath_generator,
                                                 output_types=tf.string,
                                                 output_shapes=())

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, output_labels))
        dataset = dataset.batch(self.shard_count_per_sample)
        dataset = dataset.map(self.join_shards)
        dataset = dataset.map(self.normalize_batch)
        dataset = dataset.map(self.split_batch_io)
        # dataset = dataset.batch(batch_size)

        return dataset

    def get_one_shot_iterator(self, batch_size: int, output_labels: bool):
        iterators = self._test_iterators if output_labels else self._train_iterators
        if batch_size not in iterators:
            tf_dataset = self.test_tf_dataset if output_labels else self.train_tf_dataset
            iterators[batch_size] = tf_dataset.batch(batch_size).make_one_shot_iterator()

        return iterators[batch_size]

    def get_batch(self, batch_size: int, output_labels: bool):
        iterator = self.get_one_shot_iterator(batch_size, output_labels)
        session = get_session()
        batch = session.run(iterator.get_next())
        return batch

    # region Properties
    @property
    def train_tf_dataset(self):
        if self._train_tf_dataset is None:
            self._train_tf_dataset = self.make_tf_dataset(output_labels=False)
        return self._train_tf_dataset

    @property
    def test_tf_dataset(self):
        if self._test_tf_dataset is None:
            self._test_tf_dataset = self.make_tf_dataset(output_labels=True)
        return self._test_tf_dataset

    @property
    def shard_count_per_sample(self) -> int:
        return self.config.shard_count_per_sample

    @property
    def modalities(self) -> ModalityCollection:
        return self.config.modalities
    # endregion


def main():
    # TODO : Put kitchen sink code here when done
    pass


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
