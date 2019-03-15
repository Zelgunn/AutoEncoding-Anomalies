import tensorflow as tf
import cv2
import numpy as np
import os
import cProfile


class EmolyTFRecordBuilder(object):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def list_videos_filenames(self):
        return os.listdir(self.videos_folder)

    @property
    def videos_folder(self):
        return os.path.join(self.dataset_path, "video")


# emoly_tf_record_builder = EmolyTFRecordBuilder("../datasets/emoly")
# print(emoly_tf_record_builder.list_videos_filenames())

# tf.enable_eager_execution()


def something():
    base = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\tmp\file_{}.csv"
    filenames = [base.format(i) for i in range(10)]

    filepath_dataset = tf.data.Dataset.list_files(filenames, seed=42).repeat(500)

    n_readers = 5
    dataset = filepath_dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).prefetch(1000),
        cycle_length=n_readers, num_parallel_calls=5, block_length=1)

    for line in dataset.batch(4):
        print(line.numpy())


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _bytes_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


train_filepath = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped2\Train.tfrecord"
train_filepath_2 = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped2\Train_2.tfrecord"


def convert_train():
    ucsd_filepath = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped2\Train.npz"
    video = np.load(ucsd_filepath)["videos"]
    print("Loaded")
    video = np.concatenate(video) * 255
    # video = video[0] * 255

    with tf.python_io.TFRecordWriter(train_filepath) as writer:
        encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes()) for frame in video]
        features = {"video": _bytes_list_feature(encoded_frames)}

        tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tfrecord_example.SerializeToString())
    print("Saved")


def convert_train_2():
    ucsd_filepath = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped2\Train.npz"
    video = np.load(ucsd_filepath)["videos"]
    print("Loaded")
    video = np.concatenate(video) * 255
    # video = video[0] * 255

    with tf.python_io.TFRecordWriter(train_filepath_2) as writer:
        features = {}
        for i in range(len(video)):
            ret, buffer = cv2.imencode(".jpg", video[i])
            features["frames/{:04d}".format(i)] = _bytes_feature(tf.compat.as_bytes(buffer.tobytes()))

        tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tfrecord_example.SerializeToString())
    print("Saved")


SEQ_NUM_FRAMES = 256


def parse_function(serialized_example):
    features = {"video": tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_single_example(serialized_example, features)

    random_offset = tf.random_uniform(shape=(), minval=0, maxval=2550 - SEQ_NUM_FRAMES, dtype=tf.int64)
    offsets = tf.range(random_offset, random_offset + SEQ_NUM_FRAMES)

    images = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(parsed_features["video"].values[i]), tf.float32), offsets,
                       dtype=tf.float32)
    return images


def parse_function_2(serialized_example):
    offset = np.random.randint(2550 - SEQ_NUM_FRAMES, dtype=np.int64)
    array = tf.TensorArray(dtype=tf.float32, size=SEQ_NUM_FRAMES)
    for i in range(SEQ_NUM_FRAMES):
        idx = "frames/{:04d}".format(i + offset)
        features = {idx: tf.VarLenFeature(tf.string)}
        parsed_features = tf.parse_single_example(serialized_example, features)
        image = tf.cast(tf.image.decode_jpeg(parsed_features[idx].values[0]), tf.float32)
        array = array.write(i, image)
    return array.stack()


def decode():
    import time
    dataset = tf.data.TFRecordDataset(train_filepath).repeat(32)

    dataset = dataset.map(lambda x: parse_function(x))
    dataset = dataset.batch(32)
    tfrecord_iterator = dataset.make_one_shot_iterator()

    session = tf.Session()
    t0 = time.time()
    vid = session.run(tfrecord_iterator.get_next())
    print(time.time() - t0)
    return vid


def decode_2():
    import time
    dataset = tf.data.TFRecordDataset(train_filepath_2).repeat(32)

    dataset = dataset.map(lambda x: parse_function_2(x))
    dataset = dataset.batch(32)
    tfrecord_iterator = dataset.make_one_shot_iterator()

    session = tf.Session()
    t0 = time.time()
    vid = session.run(tfrecord_iterator.get_next())
    print(time.time() - t0)
    return vid


# convert_train()
# convert_train_2()
decode()
print("=====" * 20)
decode_2()
