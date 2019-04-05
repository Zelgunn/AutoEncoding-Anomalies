import tensorflow as tf
import os
import cv2

tf.enable_eager_execution()


def parse_example(serialized_example):
    features = {"raw_video": tf.VarLenFeature(tf.string),
                "flow_x": tf.VarLenFeature(tf.string),
                "flow_y": tf.VarLenFeature(tf.string),
                "dog": tf.VarLenFeature(tf.string)}
    parsed_features = tf.parse_single_example(serialized_example, features)
    encoded_features = [parsed_features[mod].values for mod in features]
    shard_size = tf.shape(encoded_features[0])[0]
    images = [tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_shard[i]), tf.float32),
                        tf.range(shard_size),
                        dtype=tf.float32) for encoded_shard in encoded_features]
    return images


def main():
    path = r"C:\Users\Degva\Documents\_PhD\Tensorflow\datasets\subway\exit\Train"
    paths = os.listdir(path)

    paths = [p for p in paths if len(p) == 17]
    paths = [os.path.join(path, p) for p in paths]
    dataset = tf.data.TFRecordDataset(paths)
    dataset = dataset.map(parse_example)

    for batch in dataset.take(8):
        raw_video, flow_x, flow_y, dog = batch

        raw_video = raw_video.numpy()
        flow_x = flow_x.numpy()
        flow_y = flow_y.numpy()
        dog = dog.numpy()

        raw_video /= 255
        flow_x /= 5
        flow_y /= 255
        dog /= 5

        for i in range(32):
            cv2.imshow("raw_video", cv2.resize(raw_video[i], (256, 256)))

            # TODO : flow and dog are badly recorded
            # -> JPEG Encoder expects values between 0 and 255 and only allows 255 different values :(
            cv2.imshow("flow_x", cv2.resize(flow_x[i], (256, 256)))
            cv2.imshow("flow_y", cv2.resize(flow_y[i], (256, 256)))
            cv2.imshow("dog", cv2.resize(dog[i], (256, 256)))
            cv2.waitKey(40)


if __name__ == "__main__":
    main()
