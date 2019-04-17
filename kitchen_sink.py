# import tensorflow as tf
# import numpy as np
# import os
# import cv2
# from typing import List
# from tensorflow.python.framework import errors

from datasets.loaders import DatasetConfig, SubsetLoader
from modalities import ModalityShape, RawVideo, OpticalFlow  # , DoG


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
    dataset = loader.tf_dataset.batch(6)

    model = make_test_model()
    model.train_on_batch(dataset)
    print(dataset)
    exit()


if __name__ == "__main__":
    # tf.enable_eager_execution()
    main()
