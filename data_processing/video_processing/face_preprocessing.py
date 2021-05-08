import tensorflow as tf


def extract_faces(video: tf.Tensor, face_coord: tf.Tensor):
    _, height, width, _ = tf.unstack(tf.shape(video))
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    face_coord.set_shape((None, 4))
    face_coord = tf.clip_by_value(face_coord, 0.0, 1.0)
    offset_height, target_height, offset_width, target_width = tf.unstack(face_coord, axis=-1)

    offset_height = tf.reduce_min(offset_height)
    offset_width = tf.reduce_min(offset_width)
    target_height = tf.reduce_max(target_height)
    target_width = tf.reduce_max(target_width)

    target_height = tf.cast(height * (target_height - offset_height), tf.int32)
    target_width = tf.cast(width * (target_width - offset_width), tf.int32)
    offset_height = tf.cast(height * offset_height, tf.int32)
    offset_width = tf.cast(width * offset_width, tf.int32)

    video = tf.image.crop_to_bounding_box(video, offset_height, offset_width, target_height, target_width)

    return video
