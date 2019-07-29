import tensorflow as tf
import numpy as np
import os
import copy
from typing import Dict, Tuple, Optional, List

from datasets.loaders import DatasetConfig
from modalities import Modality, ModalityCollection, Pattern
from modalities import RawVideo, Landmarks
from modalities import MelSpectrogram
from utils.misc_utils import int_ceil, int_floor


# TODO : Reorder functions
class SubsetLoader(object):
    def __init__(self,
                 config: DatasetConfig,
                 subset_name: str):
        self.config = config
        self.check_unsupported_shard_sizes()

        self.subset_name = subset_name
        self.subset_files = {folder: sorted(files)
                             for folder, files in config.list_subset_tfrecords(subset_name).items()}
        self.subset_folders = list(self.subset_files.keys())

        self._train_tf_dataset: Optional[tf.data.Dataset] = None
        self._test_tf_dataset: Optional[tf.data.Dataset] = None

    def check_unsupported_shard_sizes(self):
        for modality in self.config.modalities:
            if isinstance(modality, MelSpectrogram):
                # For MelSpectrogram, max_shard_size is always equal to initial_shard_size
                continue

            shard_size = self.config.get_modality_shard_size(modality)
            max_shard_size = int_ceil(shard_size)
            initial_shard_size = int_floor(shard_size)
            if max_shard_size != initial_shard_size:
                raise ValueError("max_shard_size != initial_shard_size : "
                                 "SubsetLoader doesn't support this case yet.")

    # region Loading methods (parsing, decoding, fusing, normalization, splitting, ...)
    def parse_shard(self, serialized_examples, pattern: Pattern):
        records_count = self.modalities_per_sample(pattern)
        serialized_examples.set_shape(records_count)

        features_decoded, modalities_shard_size = {}, {}

        for i, modality_type in enumerate(pattern.modality_types):
            modality = self.modalities[modality_type]
            modality_id = modality.id()
            modality_features = modality.tfrecord_features()
            modality_example = serialized_examples[i]

            parsed_features = tf.io.parse_single_example(modality_example, modality_features)

            decoded_modality = modality.decode_from_tfrecord_feature(parsed_features)
            decoded_modality, modality_size = self.pad_modality_if_needed(modality, decoded_modality)

            features_decoded[modality_id] = decoded_modality
            modalities_shard_size[modality_id] = modality_size

        if pattern.contains_labels:
            labels_features = {"labels": tf.io.VarLenFeature(tf.float32)}
            labels_example = serialized_examples[-1]

            parsed_features = tf.io.parse_single_example(labels_example, labels_features)

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
                    pattern: Pattern,
                    length_map_function=max):

        joint_shards = {}
        labels_range = None
        labels_offset = None

        for modality in self.modalities:
            modality_type = type(modality)
            if modality_type not in pattern.as_dict:
                continue

            modality_id = modality.id()
            modality_shards = shards[modality_id]
            modality_shard_sizes = shard_sizes[modality_id]

            with tf.name_scope(modality_id):
                total_size = tf.cast(tf.reduce_sum(modality_shard_sizes), tf.int32, name="total_shard_size")

                # modality_sample_size = self.modalities[modality_type].io_shape.output_length
                modality_load_infos = pattern.as_dict[modality_type]
                modality_sample_size = length_map_function([load_info.length for load_info in modality_load_infos])
                modality_sample_size_op = tf.constant(modality_sample_size, name="modality_sample_size")

                modality_effective_size = modality_shard_sizes[0]
                modality_offset_range = tf.minimum(modality_effective_size, total_size - modality_sample_size_op)
                modality_offset_range = tf.cast(modality_offset_range, tf.float32)
                modality_offset = tf.cast(offset * modality_offset_range, tf.int32, name="offset")

                modality_shards_shape = tf.shape(modality_shards, name="modality_shard_shape")
                modality_shards_shape.set_shape(modality.rank() + 1)
                modality_shards_shape = tf.unstack(modality_shards_shape, name="unstack_modality_shape")
                shards_per_sample, modality_size, *modality_shape = modality_shards_shape
                modality_shards_shape = [shards_per_sample * modality_size, *modality_shape]
                modality_shards = tf.reshape(modality_shards, modality_shards_shape, "concatenate_shards")

                modality_shards = modality_shards[modality_offset:modality_offset + modality_sample_size_op]
                joint_shards[modality_id] = modality_shards

            if "labels" in shards and labels_range is None:
                labels_range = tf.cast(modality_sample_size_op, tf.float32) / tf.cast(total_size, tf.float32)
                size_ratio = tf.cast(modality_effective_size, tf.float32) / tf.cast(total_size, tf.float32)
                labels_offset = offset * size_ratio

        if "labels" in shards:
            labels: tf.Tensor = shards["labels"]

            shards_per_sample = tf.cast(shards_per_sample, tf.float32)
            shard_labels_offset = tf.range(shards_per_sample, dtype=tf.float32, name="shard_labels_offset")
            shard_labels_offset = tf.expand_dims(shard_labels_offset, axis=-1)

            labels = (labels + shard_labels_offset) / shards_per_sample
            labels = tf.clip_by_value(labels, labels_offset, labels_offset + labels_range)
            labels = (labels - labels_offset) / labels_range

            labels = tf.reshape(labels, shape=[-1, 2])
            joint_shards["labels"] = labels

        return joint_shards

    def normalize_batch(self, modalities: Dict[str, tf.Tensor]):
        for modality_id in modalities:
            if modality_id == "labels":
                continue

            modality_value = modalities[modality_id]
            if modality_id == RawVideo.id():
                modality_value = modality_value / tf.constant(255.0, modality_value.dtype)
            elif modality_id == MelSpectrogram.id():
                modality_value = tf.clip_by_value(modality_value, 0.0, 1.0, name="clip_mel_spectrogram")
            elif modality_id == Landmarks.id():
                modality_value = modality_value
            else:
                modality_min, modality_max = self.config.modalities_ranges[modality_id]
                modality_value = (modality_value - modality_min) / (modality_max - modality_min)
            modality_value *= (self.config.output_range[1] - self.config.output_range[0])
            modality_value += self.config.output_range[0]
            modalities[modality_id] = modality_value
        return modalities

    def add_gaussian_noise(self, *args):
        if len(args) == 2:
            inputs, outputs = args
            labels = None
        else:
            inputs, outputs, labels = args

        noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=0.1, name="gaussian_noise")
        inputs = tf.clip_by_value(inputs + noise, self.config.output_range[0], self.config.output_range[1])

        return (inputs, outputs) if (len(args) == 2) else (inputs, outputs, labels)

    # endregion

    # region Sample dataset
    @staticmethod
    def modalities_per_sample(pattern: Pattern):
        count = len(pattern.modality_types)
        if pattern.contains_labels:
            count += 1
        return count

    @staticmethod
    def shard_filepath_generator(folders: List[str],
                                 pattern: Pattern,
                                 shards_per_sample: int):
        modality_ids = list(pattern.modality_ids)
        if pattern.contains_labels:
            modality_ids.append("labels")

        while True:
            source_index = np.random.randint(len(folders))
            source_folder = folders[source_index]
            files = []
            shards_count = None
            for modality_id in modality_ids:
                folder = os.path.join(source_folder, modality_id)

                modality_files = [os.path.join(folder, file)
                                  for file in os.listdir(folder) if file.endswith(".tfrecord")]

                files.append(modality_files)
                if shards_count is None:
                    shards_count = len(modality_files)
                elif shards_count != len(modality_files):
                    raise ValueError("Modalities don't have the same number of shards in "
                                     "{}.".format(folder))

            offset = np.random.randint(shards_count - shards_per_sample + 1)
            for shard_index in range(offset, offset + shards_per_sample):
                for file_index in range(len(files)):
                    yield files[file_index][shard_index]

    def join_shards_randomly(self,
                             shards: Dict[str, tf.Tensor],
                             shard_sizes: Dict[str, tf.Tensor],
                             pattern: Pattern):
        offset = tf.random.uniform(shape=(), minval=0, maxval=1.0, dtype=tf.float32, name="offset")
        return self.join_shards(shards, shard_sizes, offset, pattern)

    def make_tf_dataset(self,
                        pattern: Pattern,
                        subset_folder: List[str] = None,
                        ):
        if subset_folder is not None:
            subset_folder = copy.copy(subset_folder)
        else:
            subset_folder = copy.copy(self.subset_folders)

        shards_per_sample = self.config.compute_shards_per_sample(pattern)

        def make_generator():
            return self.shard_filepath_generator(subset_folder, pattern, shards_per_sample)

        k = os.cpu_count()
        dataset = tf.data.Dataset.from_generator(make_generator,
                                                 output_types=tf.string,
                                                 output_shapes=())
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=k)
        dataset = dataset.batch(self.modalities_per_sample(pattern)).prefetch(1)

        dataset = dataset.map(lambda serialized_shards: self.parse_shard(serialized_shards, pattern),
                              num_parallel_calls=k)

        dataset = dataset.batch(shards_per_sample)
        dataset = dataset.map(lambda shards, shards_sizes: self.join_shards_randomly(shards,
                                                                                     shards_sizes,
                                                                                     pattern),
                              num_parallel_calls=k)

        dataset = dataset.map(self.normalize_batch, num_parallel_calls=k)
        dataset = dataset.map(pattern.apply, num_parallel_calls=k)

        return dataset

    @staticmethod
    def augment_raw_video(modalities: Dict[str, tf.Tensor]
                          ) -> Dict[str, tf.Tensor]:
        if RawVideo.id() not in modalities:
            return modalities

        raw_video = modalities[RawVideo.id()]

        # raw_video = random_video_vertical_flip(raw_video)
        raw_video = random_video_horizontal_flip(raw_video)
        raw_video = tf.image.random_hue(raw_video, max_delta=0.1)
        raw_video = tf.image.random_brightness(raw_video, max_delta=0.1)

        modalities[RawVideo.id()] = raw_video
        return modalities

    # endregion

    # region Ordered reading
    def get_batch(self, batch_size: int, pattern: Pattern):
        dataset = self.make_tf_dataset(pattern)
        dataset = dataset.batch(batch_size)
        results = None
        for results in dataset:
            break

        return results

    def make_source_filepath_generator(self,
                                       source_index: int,
                                       shards_per_sample: int,
                                       pattern: Pattern):
        modality_ids = list(pattern.modality_ids) + ["labels"]

        def generator():
            source_folder = self.subset_folders[source_index]
            files = []
            shards_count = None
            for modality_id in modality_ids:
                modality_folder = os.path.join(source_folder, modality_id)
                modality_files = [os.path.join(modality_folder, file)
                                  for file in os.listdir(modality_folder) if file.endswith(".tfrecord")]
                files.append(modality_files)

                if shards_count is None:
                    shards_count = len(modality_files)
                elif shards_count != len(modality_files):
                    raise ValueError("Modalities don't have the same number of shards in "
                                     "{}.".format(modality_folder))

            for shard_index in range(shards_count - shards_per_sample + 1):
                for i in range(shards_per_sample):
                    for modality_index in range(len(files)):
                        yield files[modality_index][shard_index + i]

        return generator

    def join_shards_ordered(self,
                            shards: Dict[str, tf.Tensor],
                            shard_sizes: Dict[str, tf.Tensor],
                            pattern: Pattern,
                            stride: int
                            ):
        with tf.name_scope("join_shards_ordered"):
            reference_modality_id = pattern.modality_ids[0]
            size = shard_sizes[reference_modality_id][0]
            stride = tf.constant(stride, tf.int32, name="stride")
            result_count = size // stride

            def loop_body(i, step_shards_arrays: Dict[str, tf.TensorArray]):
                step_offset = tf.cast(i * stride, tf.float32) / tf.cast(size - 1, tf.float32)
                step_joint_shards = self.join_shards(shards, shard_sizes, step_offset, pattern,
                                                     length_map_function=max)
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
            joint_shard = {modality_id: joint_shard[modality_id].stack(name="stack_{}".format(modality_id))
                           for modality_id in joint_shard}
            return joint_shard

    def get_source_browser(self,
                           pattern: Pattern,
                           source_index: int,
                           stride: int) -> tf.data.Dataset:
        shards_per_sample: int = self.config.compute_shards_per_sample(pattern)
        generator = self.make_source_filepath_generator(source_index, shards_per_sample, pattern)
        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=tf.string,
                                                 output_shapes=())

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.batch(self.modalities_per_sample(pattern)).prefetch(1)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, pattern))

        dataset = dataset.batch(shards_per_sample)
        dataset = dataset.map(lambda shards, shard_sizes:
                              self.join_shards_ordered(shards, shard_sizes, pattern, stride))

        dataset = dataset.unbatch()
        dataset = dataset.map(self.normalize_batch)
        dataset = dataset.map(pattern.apply)
        dataset = dataset.batch(1)

        return dataset

    # endregion

    # region Properties
    @property
    def modalities(self) -> ModalityCollection:
        return self.config.modalities

    @property
    def source_count(self) -> int:
        return len(self.subset_folders)

    # endregion

    @staticmethod
    def timestamps_labels_to_frame_labels(timestamps: np.ndarray, frame_count: int):
        # [batch_size, pairs_count, 2] => [batch_size, frame_count]
        # start, end = timestamps[:,:,0], timestamps[:,:,1]
        batch_size, timestamps_per_sample, _ = timestamps.shape
        epsilon = 1e-4
        starts = timestamps[:, :, 0]
        ends = timestamps[:, :, 1]
        labels_are_not_equal = np.abs(starts - ends) > epsilon  # [batch_size, pairs_count]

        frame_labels = np.empty(shape=[batch_size, frame_count], dtype=np.bool)

        frame_duration = 1.0 / frame_count
        for frame_id in range(frame_count):
            start_time = frame_id / frame_count
            end_time = start_time + frame_duration

            start_in = np.all([start_time >= starts, start_time <= ends], axis=0)
            end_in = np.all([end_time >= starts, end_time <= ends], axis=0)

            frame_in = np.any([start_in, end_in], axis=0)
            frame_in = np.logical_and(frame_in, labels_are_not_equal)
            frame_in = np.any(frame_in, axis=1)

            frame_labels[:, frame_id] = frame_in

        return frame_labels


# region Random video flip
def random_video_vertical_flip(video: tf.Tensor,
                               seed: int = None,
                               scope_name: str = "random_video_vertical_flip"
                               ) -> tf.Tensor:
    return random_video_flip(video, 1, seed, scope_name)


def random_video_horizontal_flip(video: tf.Tensor,
                                 seed: int = None,
                                 scope_name: str = "random_video_horizontal_flip"
                                 ) -> tf.Tensor:
    return random_video_flip(video, 2, seed, scope_name)


def random_video_flip(video: tf.Tensor,
                      flip_index: int,
                      seed: int,
                      scope_name: str
                      ) -> tf.Tensor:
    """Randomly (50% chance) flip an video along axis `flip_index`.
    Args:
        video: 5-D Tensor of shape `[batch, time, height, width, channels]` or
               4-D Tensor of shape `[time, height, width, channels]`.
        flip_index: Dimension along which to flip video. Time: 0, Vertical: 1, Horizontal: 2
        seed: A Python integer. Used to create a random seed. See `tf.set_random_seed` for behavior.
        scope_name: Name of the scope in which the ops are added.
    Returns:
        A tensor of the same type and shape as `video`.
    Raises:
        ValueError: if the shape of `video` not supported.
    """
    with tf.name_scope(scope_name) as scope:
        video = tf.convert_to_tensor(video, name="video")
        shape = video.get_shape()

        if shape.ndims == 4:
            uniform_random = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, seed=seed)
            flip_condition = tf.less(uniform_random, 0.5)
            flipped = tf.reverse(video, [flip_index])
            outputs = tf.cond(pred=flip_condition,
                              true_fn=lambda: flipped,
                              false_fn=lambda: video,
                              name=scope)

        elif shape.ndims == 5:
            batch_size = tf.shape(video)[0]
            uniform_random = tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0, seed=seed)
            uniform_random = tf.reshape(uniform_random, [batch_size, 1, 1, 1, 1])
            flips = tf.round(uniform_random)
            flips = tf.cast(flips, video.dtype)
            flipped = tf.reverse(video, [flip_index + 1])
            outputs = flips * flipped + (1.0 - flips) * video

        else:
            raise ValueError("`video` must have either 4 or 5 dimensions but has {} dimensions.".format(shape.ndims))

        return outputs


# endregion

def main():
    from modalities import ModalityLoadInfo, Pattern
    from modalities import RawAudio
    from modalities import MelSpectrogram

    audio_length = int(48000 * 1.28)
    nfft = 52

    pattern = Pattern(
        (
            ModalityLoadInfo(Landmarks, 32, (32, 136)),
            ModalityLoadInfo(MelSpectrogram, nfft, (nfft, 100))
        ),
        (
            ModalityLoadInfo(Landmarks, 128, (128, 136)),
            ModalityLoadInfo(RawAudio, audio_length, (audio_length, 2))
        )
    )
    config = DatasetConfig(tfrecords_config_folder="C:/datasets/emoly_split",
                           output_range=(0.0, 1.0)
                           )

    loader = SubsetLoader(config, "Test")
    dataset = loader.make_tf_dataset(pattern)
    print(dataset)


if __name__ == "__main__":
    main()
