import tensorflow as tf
from tensorflow.python.ops import summary_op_util
import numpy as np


# Inspired by : https://github.com/alexlee-gk
def encode_gif(images: np.ndarray,
               fps: float or int):
    """Encodes numpy images into gif string.

    Args:
      images: A 4-D `uint8` `np.array` of shape `[time, height, width, channels]` where `channels` is 1 or 3.
      fps: frames per second of the animation

    Returns:
      The encoded gif string.

    Raises:
      IOError: If the ffmpeg command returns an error.
    """

    from subprocess import Popen, PIPE

    height, width, channels = np.shape(images)[1:]
    assert channels in [1, 3]
    channels_name = "gray" if channels is 1 else "rgb24"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-r", "{0:.02f}".format(fps),
        "-s", "{0}x{1}".format(width, height),
        "-pix_fmt", channels_name,
        "-i", "-",
        "-filter_complex", "[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse",
        "-r", "{0:.02f}".format(fps),
        "-f", "gif",
        "-"]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    for image in images:
        process.stdin.write(image.tostring())
    out, error = process.communicate()

    if process.returncode:
        error = '\n'.join([' '.join(cmd), error.decode('utf8')])
        raise IOError(error)

    del process
    return out


# Inspired by : https://github.com/alexlee-gk
def py_gif_summary(tag: bytes or str,
                   images: np.ndarray,
                   max_outputs: int,
                   fps: float or int):
    """Outputs a `Summary` protocol buffer with gif animations.

    Args:
      tag: Name of the summary.
      images: A 5-D `uint8` `np.array` of shape `[batch_size, time, height, width, channels]`
        where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation

    Returns:
      The serialized `Summary` protocol buffer.

    Raises:
      ValueError: If `images` is not a 5-D `uint8` array with 1 or 3 channels.
    """

    images = np.asarray(images)

    # region Image dtype/rank/channels check
    if images.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 for gif summary.")

    if images.ndim != 5:
        raise ValueError("Tensor must be 5-D for gif summary.")

    batch_size, _, height, width, channels = images.shape
    if channels not in (1, 3):
        raise ValueError("Tensors must have 1 or 3 channels for gif summary.")
    # endregion

    if isinstance(tag, bytes):
        tag = tag.decode("utf-8")

    summary = tf.Summary()
    batch_size = min(batch_size, max_outputs)

    for i in range(batch_size):
        ith_image_summary = tf.Summary.Image()
        ith_image_summary.height = height
        ith_image_summary.width = width
        ith_image_summary.colorspace = channels
        ith_image_summary.encoded_image_string = encode_gif(images[i], fps)

        summary_tag = "{}/gif".format(tag) if (batch_size == 1) else "{}/gif/{}".format(tag, i)

        summary.value.add(tag=summary_tag, image=ith_image_summary)

    summary_string = summary.SerializeToString()
    return summary_string


# Inspired by : https://github.com/alexlee-gk
def gif_summary(name: str,
                image_tensor: tf.Tensor,
                max_outputs: int,
                fps: float or int,
                collections: str = None,
                family: str = None):
    """Outputs a `Summary` protocol buffer with gif animations.

    Args:
      name: Name of the summary.
      image_tensor: A 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width, channels]`
        where `channels` is 1 or 3.
      max_outputs: Max number of batch elements to generate gifs for.
      fps: frames per second of the animation
      collections: Optional - List of tf.GraphKeys. The collections to add the summary to.
          Defaults to [tf.GraphKeys.SUMMARIES]
      family: Optional - If provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.
    """

    image_tensor = tf.convert_to_tensor(image_tensor)
    if summary_op_util.skip_summary():
        return tf.constant("")

    with summary_op_util.summary_scope(name, family, values=[image_tensor]) as (tag, scope):
        summary_inputs = [tag, image_tensor, max_outputs, fps]
        summary_value = tf.py_func(py_gif_summary, summary_inputs, tf.string, stateful=False, name=scope)
        summary_op_util.collect(summary_value, collections, [tf.GraphKeys.SUMMARIES])

    return summary_value


def image_summary(name: str,
                  image_tensor: tf.Tensor,
                  max_outputs: int,
                  fps: float or int,
                  collections: str = None,
                  family: str = None):
    """Outputs a `Summary` protocol buffer with either one image or a gif, depending on the rank of image_tensor.

        Args:
            name: Name of the summary.
            image_tensor: A 4-D `uint8` `Tensor` of shape `[batch_size, height, width, channels]`
                or a 5-D `uint8` `Tensor` of shape `[batch_size, time, height, width, channels]`
                where `channels` is 1 or 3.
            max_outputs: Max number of batch elements to generate gifs for.
            fps: Frames per second of the animation
            collections: Optional - List of tf.GraphKeys. The collections to add the summary to.
                Defaults to [tf.GraphKeys.SUMMARIES]
            family: Optional - If provided, used as the prefix of the summary tag name,
                which controls the tab name used for display on Tensorboard.

        Returns:
            A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.
    """
    rank = image_tensor.shape.ndims
    assert rank in [4, 5]

    if rank == 4:
        return tf.summary.image(name, image_tensor, max_outputs, collections, family)
    else:
        return gif_summary(name, image_tensor, max_outputs, fps, collections, family)
