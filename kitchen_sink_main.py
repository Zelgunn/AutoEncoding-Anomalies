from kitchen_sink.audioset_ebl3 import main

import tensorflow as tf
import cv2


def tmp_main():
    input_image = tf.range(128 * 128)
    input_image = tf.reshape(input_image, [128, 128, 1])
    input_image = tf.cast(input_image, tf.float32) / (128 * 128 - 1)
    input_image *= tf.transpose(input_image, perm=[1, 0, 2])
    input_image = tf.expand_dims(input_image, axis=0)
    input_image = tf.tile(input_image, multiples=[3, 1, 1, 3])
    tmp = tf.eye(3)
    tmp = tf.reshape(tmp, [3, 1, 1, 3])
    input_image *= tmp

    output_image = tf.image.extract_patches(input_image,
                                            sizes=(1, 32, 32, 1),
                                            strides=(1, 32, 32, 1),
                                            rates=(1, 1, 1, 1),
                                            padding="SAME")

    # output_image = tf.reshape(output_image, [16 * 3, 32, 32, 3])
    # output_image = tf.reshape(output_image, [3, 1, 1, 128 * 128 * 3])
    output_image = tf.nn.depth_to_space(output_image, 32)
    output_image = tf.reshape(output_image, [3, 128, 128, 3])

    input_image_np = input_image.numpy()[0]
    cv2.imshow("input_image_np", cv2.resize(input_image_np, (512, 512)))
    output_image_np = output_image.numpy()[0]
    cv2.imshow("output_image_np", cv2.resize(output_image_np, (512, 512)))

    cv2.waitKey(0)


if __name__ == "__main__":
    tmp_main()
    exit()
    main()
