import tensorflow as tf
import os
import cv2

from datasets.loaders import IOShape, DatasetConfig, SubsetLoader


def parse_example(serialized_example):
    features = {"raw_video": tf.VarLenFeature(tf.string),
                "flow": tf.VarLenFeature(tf.string),
                "flow_shape": tf.FixedLenFeature([4], dtype=tf.int64),
                "dog": tf.VarLenFeature(tf.string),
                "dog_shape": tf.FixedLenFeature([4], dtype=tf.int64),
                "labels": tf.VarLenFeature(tf.float32)}

    parsed_features = tf.parse_single_example(serialized_example, features)
    raw_video = parsed_features["raw_video"].values
    flow = parsed_features["flow"].values
    dog = parsed_features["dog"].values
    labels = parsed_features["labels"].values

    shard_size = tf.shape(raw_video)[0]
    raw_video = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(raw_video[i]), tf.float32),
                          tf.range(shard_size),
                          dtype=tf.float32)

    flow = tf.decode_raw(flow, tf.float16)
    flow_shape = parsed_features["flow_shape"]
    flow = tf.reshape(flow, flow_shape)

    dog = tf.decode_raw(dog, tf.float16)
    dog_shape = parsed_features["dog_shape"]
    dog = tf.reshape(dog, dog_shape)

    return raw_video, flow, dog, labels


def get_ucsd_paths():
    all_paths = []
    base_path = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped2\Test\Test{:03d}"
    for i in range(12):
        ith_path = base_path.format(i + 1)
        paths = os.listdir(ith_path)
        paths = [path for path in paths if path.endswith(".tfrecord")]
        paths = [os.path.join(ith_path, p) for p in paths]
        all_paths += paths
    return all_paths


def get_subway_paths():
    path = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\subway\exit\Test"
    paths = os.listdir(path)
    paths = [path for path in paths if path.endswith(".tfrecord")]
    paths = [os.path.join(path, p) for p in paths]
    return paths


def read_and_display_dataset():
    video_io_shape = IOShape(input_shape=(16, 128, 128, 1),
                             output_shape=(32, 128, 128, 1))
    config = DatasetConfig(tfrecords_config_folder="../datasets/ucsd/ped2",
                           modalities_io_shapes=
                           {
                               "raw_video": video_io_shape,
                               "flow": video_io_shape,
                               "dog": video_io_shape
                           })
    paths = config.list_subset_tfrecords("Train")
    paths = [os.path.join(folder, filename)
             for folder in paths
             for filename in paths[folder]]

    dataset = tf.data.TFRecordDataset(paths)
    dataset = dataset.map(parse_example)

    for batch in dataset:
        raw_video, flow, dog, labels = batch

        raw_video = raw_video.numpy()
        flow = flow.numpy().astype("float32")
        flow_x = flow[:, :, :, 0]
        flow_y = flow[:, :, :, 1]
        dog = dog.numpy().astype("float32")
        labels = labels.numpy()

        raw_video /= 255
        flow_x /= 16
        flow_y /= 360
        dog /= dog.max()

        for i in range(len(raw_video)):
            raw_frame = raw_video[i]
            # raw_frame = np.tile(raw_frame, 3)

            if len(labels) > 0:
                anomaly_start = int(labels[0] * len(raw_video))
                anomaly_end = int(labels[1] * len(raw_video))

                if anomaly_start <= i <= anomaly_end:
                    raw_frame[:, :, :2] *= 0.5

            cv2.imshow("raw_video", cv2.resize(raw_frame, (256, 256)))

            cv2.imshow("flow_x", cv2.resize(flow_x[i], (256, 256)))
            cv2.imshow("flow_y", cv2.resize(flow_y[i], (256, 256)))
            cv2.imshow("dog", cv2.resize(dog[i], (256, 256)))
            cv2.waitKey(10)


def main():
    read_and_display_dataset()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
