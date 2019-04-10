import tensorflow as tf
import numpy as np
import os

from datasets.loaders import DatasetConfig

MODALITIES_PARSE_OPS = {"raw_video": tf.VarLenFeature(tf.string),
                        "flow_x": tf.VarLenFeature(tf.string),
                        "flow_y": tf.VarLenFeature(tf.string),
                        "dog": tf.VarLenFeature(tf.string),
                        "labels": tf.VarLenFeature(tf.float32)}


def get_shard_count(sample_length, shard_size):
    shard_count = 1 + np.ceil((sample_length - 1) / shard_size).astype(np.int)
    return max(1, shard_count)


class SubsetLoader(object):
    def __init__(self,
                 config: DatasetConfig,
                 subset_name: str):
        self.config = config
        self.modalities = list(config.tfrecords_modalities.keys())
        self.subset_name = subset_name
        self.subset_files = config.list_subset_tfrecords(subset_name)
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
        modalities = self.modalities + ["labels"] if output_labels else self.modalities
        features = {}
        for modality in modalities:
            features[modality] = MODALITIES_PARSE_OPS[modality]

        parsed_features = tf.parse_single_example(serialized_example, features)

        # TODO : Make dict for modalities shard sizes

        parsed_modalities = {}

        # TODO : Auto map modality -> procedure
        # parsed_modalities = {modality : procedure(parsed_features)}
        # parsed_modalities_sizes = {modality : size(parsed_features)}
        if "raw_video" in modalities:
            encoded_raw_video = parsed_features["raw_video"].values
            max_video_shard_size = self.config.get_modality_max_shard_size("raw_video")
            raw_video = self.decode_raw_video(encoded_raw_video, max_video_shard_size)
            parsed_modalities["raw_video"] = raw_video

        if "flow" in modalities:
            flow = decode_raw(parsed_features, "flow", tf.float16)
            parsed_modalities["flow"] = flow

        if "dog" in modalities:
            dog = decode_raw(parsed_features, "dog", tf.float16)
            parsed_modalities["dog"] = dog

        return parsed_modalities, parsed_modalities_sizes

    def decode_raw_video(self, encoded_raw_video, max_shard_size: int):
        # TODO : Type encoded_raw_video
        print(type(encoded_raw_video))
        exit()

        raw_video_shard_size = tf.shape(encoded_raw_video)[0]
        raw_video = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_raw_video[i]), tf.float32),
                              tf.range(raw_video_shard_size),
                              dtype=tf.float32)
        raw_video_pad_size = max_shard_size - raw_video_shard_size

        def images_padded():
            paddings = [[0, raw_video_pad_size], [0, 0], [0, 0], [0, 0]]
            return tf.pad(raw_video, paddings)

        def images_identity():
            return raw_video

        raw_video = tf.cond(pred=raw_video_pad_size > 0,
                            true_fn=images_padded,
                            false_fn=images_identity)

        return raw_video

    def join_shards_modality(self,
                             shards,
                             shards_length,
                             modality: str):
        # TODO : Type shards and shards_length
        print(type(shards))
        print(type(shards_length))
        exit()
        total_length = tf.cast(tf.reduce_sum(shards_length), tf.int64)
        modality_sample_length = self.config.get_modality_sample_length(modality)
        length_delta = total_length - modality_sample_length
        offset = tf.random.uniform(shape=(), minval=0, maxval=length_delta, dtype=tf.int64)
        shard_count, shard_size, height, width, channels = tf.unstack(tf.shape(shards))
        shards = tf.reshape(shards, [shard_count * shard_size, height, width, channels])
        shards = shards[offset:offset + modality_sample_length]
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

    @property
    def shard_count_per_sample(self) -> int:
        return self.config.shard_count_per_sample


def decode_raw(parsed_features, modality: str, dtype: tf.dtypes.DType):
    # TODO : Type parsed_features
    print(type(parsed_features))
    exit()
    encoded_modality = parsed_features[modality].values
    modality_shape = parsed_features[modality + "_shape"]

    decoded_modality = tf.decode_raw(encoded_modality, dtype)
    decoded_modality = tf.reshape(decoded_modality, modality_shape)
    return decoded_modality


def main():
    # TODO : Put kitchen sink code here when done
    pass


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
