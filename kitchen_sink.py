import tensorflow as tf
import cv2
import numpy as np
from datasets import VideoDatasetV2

tf.enable_eager_execution()


# region Helpers
def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _bytes_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


# endregion


train_filepath = r"..\datasets\ucsd\ped2\Train_{}.tfrecord"


def convert_function(video):
    encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in video]
    features = {"video": _bytes_list_feature(encoded_frames)}

    return features


def encode():
    ucsd_filepath = r"..\datasets\ucsd\ped2\Train.npz"
    video = np.load(ucsd_filepath)["videos"]
    video = np.concatenate(video) * 255

    batch_count = np.ceil(len(video) / REC_BATCH_SIZE).astype(np.int64)

    for i in range(batch_count):
        filepath = train_filepath.format("2_{}".format(i))
        with tf.python_io.TFRecordWriter(filepath) as writer:
            features = convert_function(video[i * REC_BATCH_SIZE:(i + 1) * REC_BATCH_SIZE])

            tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tfrecord_example.SerializeToString())


SEQ_NUM_FRAMES = 32
REC_BATCH_SIZE = 32


def parse_shard(serialized_example):
    features = {"video": tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_single_example(serialized_example, features)

    encoded_shard = parsed_features["video"].values
    shard_length = tf.shape(encoded_shard)[0]

    images = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_shard[i]), tf.float32),
                       tf.range(shard_length),
                       dtype=tf.float32)

    def images_padded():
        paddings = [[0, REC_BATCH_SIZE - shard_length], [0, 0], [0, 0], [0, 0]]
        return tf.pad(images, paddings)

    def images_identity():
        return images

    images = tf.cond(shard_length < REC_BATCH_SIZE,
                     images_padded,
                     images_identity)

    return images, shard_length


def join_shards(shards, shards_length):
    total_length = tf.cast(tf.reduce_sum(shards_length), tf.int64)
    offset = tf.random.uniform(shape=(), minval=0, maxval=total_length - SEQ_NUM_FRAMES, dtype=tf.int64)
    shard_count, shard_size, height, width, channels = tf.unstack(tf.shape(shards))
    shards = tf.reshape(shards, [shard_count * shard_size, height, width, channels])
    shards = shards[offset:offset + SEQ_NUM_FRAMES]
    return shards


# def decode():
#     steps = 32
#
#     dataset = tf.data.TFRecordDataset(train_filepath).repeat(steps)
#
#     dataset = dataset.map(lambda x: parse_function(x))
#     dataset = dataset.batch(1)
#     tfrecord_iterator = dataset.make_one_shot_iterator()
#
#     session = tf.Session()
#
#     profiler = tf.profiler.Profiler()
#
#     total_time = 0
#     for i in range(steps):
#         options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#         run_metadata = tf.RunMetadata()
#
#         t0 = time.time()
#         x = session.run(tfrecord_iterator.get_next(),
#                         options=options,
#                         run_metadata=run_metadata)
#         t1 = time.time()
#         print(i, t1 - t0, x.shape)
#         total_time += t1 - t0
#
#         profiler.add_step(i, run_metadata)
#         # profiler_options = tf.profiler.ProfileOptionBuilder.time_and_memory()
#         # profiler.profile_operations(profiler_options)
#
#     advices = profiler.advise({
#         'ExpensiveOperationChecker': {},
#         'AcceleratorUtilizationChecker': {},
#         'JobChecker': {},  # Only available internally.
#         'OperationChecker': {},
#     })
#
#     print(advices)
#     print("Total time :", total_time)
