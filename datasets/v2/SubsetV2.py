import tensorflow as tf
import numpy as np
import os

from datasets import DatasetConfigV2

MODALITIES_PARSE_OPS = {"raw_video": tf.VarLenFeature(tf.string),
                        "flow_x": tf.VarLenFeature(tf.string),
                        "flow_y": tf.VarLenFeature(tf.string),
                        "dog": tf.VarLenFeature(tf.string),
                        "labels": tf.VarLenFeature(tf.float32)}


def get_shard_count(sample_length, shard_size):
    shard_count = 1 + np.ceil((sample_length - 1) / shard_size).astype(np.int)
    return max(1, shard_count)


# TODO : Rename class to SubsetBuilder or something like that (not Subset/Dataset - taken by TF)
class SubsetV2(object):
    def __init__(self,
                 dataset_path: str,
                 config: DatasetConfigV2):
        self.dataset_path = dataset_path
        self.shard_count_per_sample = get_shard_count(config.sample_length, config.shard_size)
        self.config = config

        # TODO : records_dict should come from Dataset
        self.records_dict = get_video_shard_dict(dataset_path)["Train"]
        self.sources_filepath = list(self.records_dict.keys())

        # TODO : Load available modalities list from header, remove all un-available modalities
        self.modalities = ["video"]

    def shard_filepath_generator(self):
        while True:
            source_index = np.random.randint(len(self.sources_filepath))
            source_path = self.sources_filepath[source_index]
            shards = self.records_dict[source_path]
            offset = np.random.randint(len(shards) - self.shard_count_per_sample + 1)
            shards_filepath = [os.path.join(source_path, shard)
                               for shard in shards[offset:offset + self.shard_count_per_sample]]
            for filepath in shards_filepath:
                yield filepath

    def parse_shard(self, serialized_example, output_labels):
        modalities = self.modalities + ["labels"] if output_labels else self.modalities
        features = {}
        for modality in modalities:
            features[modality] = MODALITIES_PARSE_OPS[modality]

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

    def get_dataset_iterator(self, output_labels, batch_size):
        dataset = tf.data.Dataset.from_generator(self.shard_filepath_generator,
                                                 output_types=tf.string,
                                                 output_shapes=())

        # TODO : Remove next line
        dataset = dataset.take(512)

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(lambda serialized_shard: self.parse_shard(serialized_shard, output_labels))
        dataset = dataset.batch(self.shard_count_per_sample)
        dataset = dataset.map(self.join_shards)
        dataset = dataset.batch(batch_size)

        return dataset


# TODO : Rework for different data supports
def get_video_shard_dict(dataset_path: str):
    potential_subsets = []
    for element in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, element)):
            potential_subsets.append(element)

    subsets_dict = {}
    for potential_subset in potential_subsets:
        subset_path = os.path.join(dataset_path, potential_subset)
        for video_dir in os.listdir(subset_path):
            video_path = os.path.join(subset_path, video_dir)
            if os.path.isdir(video_path):
                video_shards = os.listdir(video_path)
                video_shards = [shard for shard in video_shards if shard.endswith(".tfrecord")]
                if len(video_shards) > 0:
                    if potential_subset in subsets_dict:
                        subsets_dict[potential_subset][video_path] = video_shards
                    else:
                        subsets_dict[potential_subset] = {video_path: video_shards}
    return subsets_dict


def main():
    # TODO : Input/Output sequence length are deprecated
    config = DatasetConfigV2(input_sequence_length=16,
                             output_sequence_length=32)

    # TODO : Get shard length from collections of length in header file
    config.shard_size = 32
    subset = SubsetV2(r"..\datasets\ucsd\ped2", config)

    dataset = subset.get_dataset_iterator(True, batch_size=16)
    for batch in dataset.take(8):
        print(batch.shape)


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
