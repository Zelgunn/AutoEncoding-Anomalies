import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from protocols.video_protocols import UCSDProtocol
from datasets.data_readers import VideoReader
from misc_utils.math_utils import standardize_from
from custom_tf_models import LED


def main():
    protocol = UCSDProtocol()
    model: LED = protocol.model
    model.load_weights(r"D:\Users\Degva\Desktop\tmp\work\LED\weights_037")

    video_reader = VideoReader(r"D:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped2\Test\Test004")
    frames = [frame for i, frame in zip(range(32), video_reader)]
    frames = tf.stack(frames, axis=0)
    frames = tf.cast(frames, tf.float32)
    frames = tf.expand_dims(frames, axis=-1)
    frames = tf.image.resize(frames, (128, 128))
    frames = standardize_from(frames, start_axis=0)
    frames = tf.expand_dims(frames, axis=0)

    # encoded_masked = model.encode(frames)
    encoded_unmasked = model.encoder(frames)

    # decoded_masked = model.decoder(encoded_masked)
    # decoded_unmasked = model.decoder(encoded_unmasked)
    #
    # save_decoded(decoded_masked, r"D:\Users\Degva\Desktop\tmp\work\LED\decoded_masked.png")
    # save_decoded(decoded_unmasked, r"D:\Users\Degva\Desktop\tmp\work\LED\decoded_unmasked.png")

    energy = model.description_energy_model(encoded_unmasked)
    mask = model.get_description_mask(energy)
    # mask = tf.reshape(mask, [2 * 8 * 8, 128])
    mask = tf.reshape(mask, [2 * 8 * 8, 128])
    mask = tf.reduce_max(mask, axis=0)

    energy = tf.reshape(energy, [2 * 8 * 8, 128])
    energy = tf.reduce_max(energy, axis=0)

    encoded_unmasked = tf.reshape(encoded_unmasked, [2 * 8 * 8, 128])
    encoded_unmasked_max = tf.reduce_max(encoded_unmasked, axis=0)
    encoded_unmasked_mean = tf.reduce_mean(encoded_unmasked, axis=0)

    data = tf.stack([mask, energy, encoded_unmasked_max, encoded_unmasked_mean], axis=-1).numpy()

    plt.plot(data)
    plt.show()


def save_decoded(x: tf.Tensor, path):
    x = normalize(x)
    x = x.numpy()[0, 15]
    cv2.imwrite(path, x * 255.0)


def normalize(x):
    x_min = tf.reduce_min(x)
    x = (x - x_min) / (tf.reduce_max(x) - x_min)
    return x


if __name__ == "__main__":
    main()
