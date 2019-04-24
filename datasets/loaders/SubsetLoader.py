import tensorflow as tf
from tensorflow.python.keras.backend import get_session
import numpy as np
import os
from typing import Dict, Tuple, Optional, Type, NamedTuple

from datasets.loaders import DatasetConfig
from modalities import Modality, ModalityCollection, RawVideo


def get_shard_count(sample_length, shard_size):
    shard_count = 1 + np.ceil((sample_length - 1) / shard_size).astype(np.int)
    return max(1, shard_count)


class BrowserIterator(NamedTuple):
    base_iterator: tf.data.Iterator
    initializer: tf.Operation
    iterator_next: Dict[str, tf.Tensor]


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

        self._source_browsers_iterators: Dict[int, BrowserIterator] = {}

    # region Loading methods (parsing, decoding, fusing, normalization, splitting, ...)
    def parse_shard(self, serialized_example, output_labels):
        features = self.modalities.get_tfrecord_features()
        if output_labels:
            features["labels"] = tf.VarLenFeature(tf.float32)

        parsed_features = tf.parse_single_example(serialized_example, features)

        modalities_shard_size, features_decoded = {}, {}

        for modality in self.modalities:
            modality_id = modality.id()
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
                                 name=modality.id() + "_paddings")
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
                    shard_sizes: Dict[str, tf.Tensor],
                    offset: tf.Tensor,
                    sample_length: str = "output_length"):

        joint_shards = {}
        labels_range = None
        labels_offset = None

        for modality in self.modalities:
            modality_type = type(modality)
            modality_id = modality.id()
            modality_shards = shards[modality_id]
            modality_shard_sizes = shard_sizes[modality_id]

            with tf.name_scope(modality_id):
                total_size = tf.cast(tf.reduce_sum(modality_shard_sizes), tf.int32, name="total_shard_size")

                if sample_length == "output_length":
                    modality_sample_size = self.modalities[modality_type].io_shape.output_length
                else:
                    modality_sample_size = self.modalities[modality_type].io_shape.input_length
                modality_sample_size = tf.constant(modality_sample_size, name="modality_sample_size")

                modality_effective_size = modality_shard_sizes[0]
                modality_offset_range = tf.minimum(modality_effective_size, total_size - modality_sample_size)
                modality_offset_range = tf.cast(modality_offset_range, tf.float32)
                modality_offset = tf.cast(offset * modality_offset_range, tf.int32, name="offset")

                modality_shards_shape = tf.unstack(tf.shape(modality_shards), name="modality_shape")
                shard_count, modality_size, *modality_shape = modality_shards_shape
                modality_shards_shape = [shard_count * modality_size, *modality_shape]
                modality_shards = tf.reshape(modality_shards, modality_shards_shape, "concatenate_shards")

                modality_shards = modality_shards[modality_offset:modality_offset + modality_sample_size]
                joint_shards[modality_id] = modality_shards

            if "labels" in shards and labels_range is None:
                labels_range = tf.cast(modality_sample_size, tf.float32) / tf.cast(total_size, tf.float32)
                size_ratio = tf.cast(modality_effective_size, tf.float32) / tf.cast(total_size, tf.float32)
                labels_offset = offset * size_ratio

        if "labels" in shards:
            labels: tf.Tensor = shards["labels"]

            shard_labels_offset = tf.range(self.shards_per_sample, dtype=tf.float32, name="shard_labels_offset")
            shard_labels_offset = tf.expand_dims(shard_labels_offset, axis=-1)

            labels = (labels + shard_labels_offset) / self.shards_per_sample
            labels = tf.clip_by_value(labels, labels_offset, labels_offset + labels_range)
            labels = (labels - labels_offset) / labels_range

            labels = tf.reshape(labels, shape=[-1, 2])
            joint_shards["labels"] = labels

        return joint_shards

    # TODO : Normalize target range according to model config (model output)
    def normalize_batch(self, modalities: Dict[str, tf.Tensor]):
        for modality_id in modalities:
            if modality_id == "labels":
                continue

            modality_value = modalities[modality_id]
            if modality_id == RawVideo.id():
                modality_value = modality_value / tf.constant(255.0, modality_value.dtype)
            else:
                modality_min, modality_max = self.config.modalities_ranges[modality_id]
                modality_value = (modality_value - modality_min) / (modality_max - modality_min)
            modality_value *= (self.config.output_range[1] - self.config.output_range[0])
            modality_value += self.config.output_range[0]
            modalities[modality_id] = modality_value
        return modalities

    def split_batch_io(self, modalities: Dict[str, tf.Tensor]):
        inputs, outputs = [], []
        labels = None
        for modality_id, modality_value in modalities.items():
            if modality_id == "labels":
                labels = modality_value
                continue

            modality_type = ModalityCollection.modality_id_to_class(modality_id)

            input_length = self.modalities[modality_type].io_shape.input_length
            output_length = self.modalities[modality_type].io_shape.output_length

            inputs.append(modality_value[:input_length])
            outputs.append(modality_value[-output_length:])

        if labels is None:
            return tuple(inputs), tuple(outputs)
        else:
            return tuple(inputs), tuple(outputs), labels

    # endregion

    # region Random sampling
    def random_shard_filepath_generator(self):
        while True:
            source_index = np.random.randint(len(self.subset_folders))
            source_folder = self.subset_folders[source_index]
            shards = self.subset_files[source_folder]
            offset = np.random.randint(len(shards) - self.shards_per_sample + 1)
            shards_filepath = [os.path.join(source_folder, shard)
                               for shard in shards[offset:offset + self.shards_per_sample]]
            for filepath in shards_filepath:
                yield filepath

    def join_shards_randomly(self,
                             shards: Dict[str, tf.Tensor],
                             shard_sizes: Dict[str, tf.Tensor]):
        offset = tf.random.uniform(shape=(), minval=0, maxval=1.0, dtype=tf.float32, name="offset")
        return self.join_shards(shards, shard_sizes, offset)

    def make_tf_dataset(self, output_labels: bool):
        dataset = tf.data.Dataset.from_generator(self.random_shard_filepath_generator,
                                                 output_types=tf.string,
                                                 output_shapes=())

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, output_labels))
        dataset = dataset.batch(self.shards_per_sample)
        dataset = dataset.map(self.join_shards_randomly)
        dataset = dataset.map(self.normalize_batch)
        dataset = dataset.map(self.split_batch_io)

        return dataset

    def get_one_shot_iterator(self, batch_size: int, output_labels: bool):
        iterators = self._test_iterators if output_labels else self._train_iterators
        if batch_size not in iterators:
            tf_dataset = self.labeled_tf_dataset if output_labels else self.tf_dataset
            iterators[batch_size] = tf_dataset.batch(batch_size).make_one_shot_iterator().get_next()

        return iterators[batch_size]

    # endregion

    def get_batch(self, batch_size: int, output_labels: bool):
        iterator = self.get_one_shot_iterator(batch_size, output_labels)
        session = get_session()
        inputs, outputs = session.run(iterator)
        return inputs, outputs

    # region Read dataset in order
    def make_source_filepath_generator(self, source_index: int):
        def generator():
            source_folder = self.subset_folders[source_index]
            shards_filepaths = [os.path.join(source_folder, shard) for shard in self.subset_files[source_folder]]
            for i in range(len(shards_filepaths) - 1):
                for filepath in shards_filepaths[i:i + 2]:
                    yield filepath

        return generator

    def join_shards_ordered(self,
                            shards: Dict[str, tf.Tensor],
                            shard_sizes: Dict[str, tf.Tensor],
                            reference_modality: Type[Modality],
                            stride: int
                            ):
        reference_modality_id = reference_modality.id()
        size = shard_sizes[reference_modality_id][0]
        stride = tf.constant(stride, tf.int32, name="stride")
        result_count = size // stride

        def loop_body(i, step_shards_arrays: Dict[str, tf.TensorArray]):
            step_offset = tf.cast(i * stride, tf.float32) / tf.cast(size - 1, tf.float32)
            step_joint_shards = self.join_shards(shards, shard_sizes, step_offset, sample_length="input_length")
            for modality_id in step_shards_arrays:
                modality = step_joint_shards[modality_id]
                step_shards_arrays[modality_id] = step_shards_arrays[modality_id].write(i, modality)
            i += 1
            return i, step_shards_arrays

        i_initializer = tf.constant(0, tf.int32)
        shards_arrays = {modality_id: tf.TensorArray(dtype=shards[modality_id].dtype, size=result_count)
                         for modality_id in shards}
        results = tf.while_loop(cond=lambda i, _: i < result_count,
                                body=loop_body,
                                loop_vars=[i_initializer, shards_arrays],
                                parallel_iterations=1)
        joint_shard: Dict[str, tf.TensorArray] = results[1]
        joint_shard = {modality_id: joint_shard[modality_id].stack() for modality_id in joint_shard}
        return joint_shard

    def get_source_browser(self,
                           source_index: int,
                           reference_modality: Type[Modality],
                           stride: int) -> tf.data.Dataset:

        dataset = tf.data.Dataset.from_generator(self.make_source_filepath_generator(source_index),
                                                 output_types=tf.string,
                                                 output_shapes=())

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, output_labels=True))
        dataset = dataset.batch(self.shards_per_sample)
        dataset = dataset.map(lambda shards, shard_sizes:
                              self.join_shards_ordered(shards, shard_sizes, reference_modality, stride))
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.map(self.normalize_batch)
        dataset = dataset.batch(1)

        return dataset

    def get_source_browser_iterator(self,
                                    source_index: int,
                                    reference_modality: Type[Modality],
                                    stride: int
                                    ):
        if source_index not in self._source_browsers_iterators:
            tf_dataset = self.get_source_browser(source_index, reference_modality, stride)
            tf_dataset = tf_dataset.map(lambda dictionary: tuple(dictionary.values()))
            base_iterator = tf_dataset.make_initializable_iterator()
            iterator_next = base_iterator.get_next()
            iterator = BrowserIterator(base_iterator=base_iterator,
                                       initializer=base_iterator.initializer,
                                       iterator_next=iterator_next)
            self._source_browsers_iterators[source_index] = iterator

        return self._source_browsers_iterators[source_index]

    # endregion

    # region Properties
    @property
    def tf_dataset(self) -> tf.data.Dataset:
        if self._train_tf_dataset is None:
            self._train_tf_dataset = self.make_tf_dataset(output_labels=False)
        return self._train_tf_dataset

    @property
    def labeled_tf_dataset(self) -> tf.data.Dataset:
        if self._test_tf_dataset is None:
            self._test_tf_dataset = self.make_tf_dataset(output_labels=True)
        return self._test_tf_dataset

    @property
    def shards_per_sample(self) -> int:
        return self.config.shard_count_per_sample

    @property
    def modalities(self) -> ModalityCollection:
        return self.config.modalities

    @property
    def source_count(self) -> int:
        return len(self.subset_folders)
    # endregion


def main():
    import cv2
    from modalities import RawVideo, OpticalFlow, ModalityShape

    tf.enable_eager_execution()

    config = DatasetConfig(tfrecords_config_folder="../datasets/ucsd/ped2",
                           modalities_io_shapes=
                           {
                               RawVideo: ModalityShape(input_shape=(16, 128, 128, 1),
                                                       output_shape=(32, 128, 128, 1)),
                               OpticalFlow: ModalityShape(input_shape=(16, 128, 128, 1),
                                                          output_shape=(32, 128, 128, 2)),
                               # DoG: video_io_shape
                           })

    loader = SubsetLoader(config, "Test")

    for source_index in range(loader.source_count):
        dataset = loader.get_source_browser(source_index, RawVideo, 1)

        for batch in dataset.batch(1000):
            raw_video = np.squeeze(batch[RawVideo.id()])

            labels = np.squeeze(batch["labels"], axis=1)
            labels_not_equal: np.ndarray = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
            labels_not_equal = np.any(labels_not_equal, axis=-1)

            for i in range(len(labels)):
                frame = raw_video[i][-1]

                if labels_not_equal[i]:
                    frame *= 0.5

                frame = cv2.resize(frame, (512, 512))
                cv2.imshow("frame", frame)
                cv2.waitKey(1000 // 25)


if __name__ == "__main__":
    main()
