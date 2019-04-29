import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend
import os
from time import time

from datasets.loaders import DatasetConfig, SubsetLoader
from modalities import ModalityShape, RawVideo


def make_test_model(input_shape):
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import Input, Conv3D, Conv3DTranspose, Flatten, Reshape, Concatenate, \
        TimeDistributed

    from KanervaMemory import Memory

    def make_encoder(encoder_input_layer):
        layer = encoder_input_layer
        layer = Conv3D(filters=16, kernel_size=3, strides=(1, 2, 2), padding="same",
                       activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                       activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                       activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                       activation="relu")(layer)
        layer = Conv3D(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                       activation="relu")(layer)

        return TimeDistributed(Flatten(), name="flatten")(layer)

    def make_decoder(decoder_input_layer, output_filters):
        layer = decoder_input_layer
        layer = Reshape([16, 4, 4, 32])(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                                activation="relu")(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                                activation="relu")(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                                activation="relu")(layer)
        layer = Conv3DTranspose(filters=32, kernel_size=3, strides=(1, 2, 2), padding="same",
                                activation="relu")(layer)
        layer = Conv3DTranspose(filters=output_filters, kernel_size=3, strides=(1, 2, 2), padding="same",
                                activation="sigmoid")(layer)
        return layer

    input_layer = Input(shape=input_shape)

    embeddings = make_encoder(input_layer)

    memory = Memory(code_size=512, memory_size=32)
    embeddings = memory(embeddings)

    decoder = make_decoder(embeddings, 1)
    predictor = make_decoder(embeddings, 1)

    output_layer = Concatenate(axis=1)([decoder, predictor])

    inputs = [input_layer]
    outputs = [output_layer]

    model = Model(inputs=inputs, outputs=outputs)

    def reconstruction_loss(y_true, y_pred):
        return backend.mean(backend.sum(backend.square(y_true - y_pred), axis=[2, 3, 4]))

    model.compile(Adam(lr=1e-4), loss=reconstruction_loss, metrics=["mse"])

    return model


def main():
    input_shape = (16, 128, 128, 1)
    output_shape = (input_shape[0] * 2, *input_shape[1:])

    config = DatasetConfig(tfrecords_config_folder="../datasets/ucsd/ped2",
                           modalities_io_shapes=
                           {
                               RawVideo: ModalityShape(input_shape=input_shape,
                                                       output_shape=output_shape),
                           },
                           output_range=(0.0, 1.0))

    loader = SubsetLoader(config, "Test")
    dataset = loader.tf_dataset.batch(32)

    model = make_test_model(input_shape=input_shape)

    log_dir = "../logs/KanervaMachine/log_{}".format(int(time()))
    os.makedirs(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, update_freq="batch")

    model.fit(dataset, steps_per_epoch=500, epochs=6, callbacks=[tensorboard])

    import cv2

    predictions = model.predict(loader.get_batch(32, output_labels=False)[0])

    k = 0
    for video in predictions:
        for frame in video:
            frame = cv2.resize(frame, (512, 512))
            cv2.imshow("frame", frame)
            cv2.waitKey(k)
            k = 40


if __name__ == "__main__":
    # import tensorflow as tf
    # tf.enable_eager_execution()
    main()
