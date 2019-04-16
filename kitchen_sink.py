import tensorflow as tf
import numpy as np
import os
import cv2
from typing import List
from tensorflow.python.framework import errors

from datasets.loaders import DatasetConfig, SubsetLoader
from modalities import ModalityShape, RawVideo, OpticalFlow, DoG


def parse_example(serialized_example):
    features = {"RawVideo": RawVideo.tfrecord_features(),
                "OpticalFlow": tf.VarLenFeature(tf.string),
                "OpticalFlow_shape": OpticalFlow.tfrecord_shape_parse_function(),
                "DoG": DoG.tfrecord_features(),
                "DoG_shape": DoG.tfrecord_shape_parse_function(),
                "labels": tf.VarLenFeature(tf.float32)}

    parsed_features = tf.parse_single_example(serialized_example, features)
    raw_video = parsed_features["RawVideo"].values
    flow = parsed_features["OpticalFlow"].values
    dog = parsed_features["DoG"].values
    labels = parsed_features["labels"].values

    shard_size = tf.shape(raw_video)[0]
    raw_video = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(raw_video[i]), tf.float32),
                          tf.range(shard_size),
                          dtype=tf.float32)

    flow = tf.decode_raw(flow, tf.float16)
    flow_shape = parsed_features["OpticalFlow_shape"]
    flow = tf.reshape(flow, flow_shape)

    dog = tf.decode_raw(dog, tf.float16)
    dog_shape = parsed_features["DoG_shape"]
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


def join_raw_labels(raw_labels: np.ndarray) -> List[np.ndarray]:
    labels = []
    for label in raw_labels:
        if label[0] != label[1]:
            labels.append(label)

    i = 0
    while i < (len(labels) - 1):
        if labels[i][1] == labels[i + 1][0]:
            labels[i] = [labels[i][0], labels[i + 1][1]]
            labels.pop(i + 1)
        else:
            i += 1
    return labels


def read_and_display_dataset():
    video_io_shape = ModalityShape(input_shape=(16, 128, 128, 1),
                                   output_shape=(32, 128, 128, 1))
    config = DatasetConfig(tfrecords_config_folder="../datasets/ucsd/ped2",
                           modalities_io_shapes=
                           {
                               RawVideo: video_io_shape,
                               OpticalFlow: video_io_shape,
                               DoG: video_io_shape
                           })

    loader = SubsetLoader(config, "Test")
    dataset = loader.make_tf_dataset(output_labels=True).batch(1)

    for batch in dataset:
        raw_video, flow, dog, raw_labels = tuple(batch.values())

        raw_video = raw_video.numpy()
        flow = flow.numpy().astype("float32")
        dog = dog.numpy().astype("float32")
        raw_labels = raw_labels.numpy()

        raw_video = np.squeeze(raw_video, axis=0)
        flow = np.squeeze(flow, axis=0)
        dog = np.squeeze(dog, axis=0)
        raw_labels = np.squeeze(raw_labels, axis=0)
        labels = join_raw_labels(raw_labels)

        flow_x = flow[:, :, :, 0]
        flow_y = flow[:, :, :, 1]

        raw_video /= 255
        flow_x /= 16
        flow_y /= 360
        dog /= dog.max()

        for i in range(len(raw_video)):
            raw_frame = raw_video[i]
            # raw_frame = np.tile(raw_frame, 3)

            if len(labels) > 0:
                anomaly_start = int(labels[0][0] * len(raw_video))
                anomaly_end = int(labels[0][1] * len(raw_video))

                if anomaly_start <= i <= anomaly_end:
                    raw_frame[:, :, :2] *= 0.5

            cv2.imshow("raw_video", cv2.resize(raw_frame, (256, 256)))
            cv2.imshow("flow_x", cv2.resize(flow_x[i], (256, 256)))
            cv2.imshow("flow_y", cv2.resize(flow_y[i], (256, 256)))
            cv2.imshow("dog", cv2.resize(dog[i], (256, 256)))
            cv2.waitKey(40)
        # cv2.waitKey(0)


def make_test_model():
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input, Conv3D, Conv3DTranspose, Flatten, Reshape, Concatenate, Dense

    def make_encoder(input_layer):
        layer = input_layer
        layer = Conv3D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same", activation="relu")(layer)

        return Flatten()(layer)

    def make_decoder(input_layer, output_filters):
        layer = input_layer
        layer = Reshape([1, 4, 4, 32])(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same", activation="relu")(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(layer)
        layer = Conv3DTranspose(filters=output_filters, kernel_size=3, strides=2, padding="same",
                                activation="sigmoid")(layer)
        return layer

    raw_video_input_layer = Input(shape=(16, 128, 128, 1))
    optical_flow_input_layer = Input(shape=(16, 128, 128, 2))

    raw_video_embeddings = make_encoder(raw_video_input_layer)
    optical_flow_embeddings = make_encoder(optical_flow_input_layer)

    embeddings = Concatenate()([raw_video_embeddings, optical_flow_embeddings])
    embeddings = Dense(units=512)(embeddings)

    raw_video_decoder = make_decoder(embeddings, 1)
    optical_flow_decoder = make_decoder(embeddings, 2)

    raw_video_predictor = make_decoder(embeddings, 1)
    optical_flow_predictor = make_decoder(embeddings, 2)

    raw_video_output_layer = Concatenate(axis=1)([raw_video_decoder, raw_video_predictor])
    optical_flow_output_layer = Concatenate(axis=1)([optical_flow_decoder, optical_flow_predictor])

    inputs = [raw_video_input_layer, optical_flow_input_layer]
    outputs = [raw_video_output_layer, optical_flow_output_layer]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile("adam", loss="mse")

    return model


def main():
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
    dataset = loader.get_source_browser(11, RawVideo, 1)
    # iterator = dataset.make_initializable_iterator()
    # iterator_next = iterator.get_next()

    for batch in dataset.batch(150):
        raw_video = np.squeeze(batch["RawVideo"])
        labels = np.squeeze(batch["labels"], axis=1)
        # labels = np.reshape(labels, [len(labels), -1])
        labels_not_equal: np.ndarray = np.abs(labels[:, :, 0] - labels[:, :, 1]) > 1e-7
        labels_not_equal = np.any(labels_not_equal, axis=-1)
        for i in range(len(labels)):
            frame = raw_video[i][-1]
            if labels_not_equal[i]:
                frame *= 0.5
            frame = cv2.resize(frame, (512, 512))
            cv2.imshow("frame", frame)
            cv2.waitKey(50)


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
