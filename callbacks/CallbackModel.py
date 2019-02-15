import tensorflow as tf
from keras.backend import get_session
from keras.utils.generic_utils import to_list
import numpy as np
from typing import List


class CallbackModel(object):
    def __init__(self,
                 inputs: tf.Tensor or List,
                 outputs: tf.Tensor or List,
                 output_is_summary=False):
        self.inputs = to_list(inputs, allow_tuple=True)
        self.outputs = to_list(outputs, allow_tuple=True)
        self.output_is_summary = output_is_summary

    def run(self, x, session=None):
        if self.output_is_summary:
            return self.run_summary(x, session)
        else:
            return self.predict(x, session)

    def predict(self, x, batch_size=32, session=None):
        x = to_list(x, allow_tuple=True)
        assert len(x) == len(self.inputs),\
            "Expected x to contain {0} inputs (received {1})".format(len(self.inputs), len(x))

        if session is None:
            session = get_session()

        total_size = len(x[0])
        batch_count = int(np.ceil(total_size / batch_size))

        results = [None] * batch_count

        for i in range(batch_count):
            start, end = i * batch_size, (i + 1) * batch_size
            values = [array[start:end] for array in x]
            feed_dict = dict(zip(self.inputs, values))
            results[i] = session.run(self.outputs, feed_dict)

        transpose_axes = [1, 0] + list(range(2, len(np.shape(results))))
        results = np.transpose(results, axes=transpose_axes)
        results = np.reshape(results, newshape=[len(self.outputs), total_size, *results.shape[3:]])

        return results

    def run_summary(self, x, session=None):
        x = to_list(x, allow_tuple=True)
        assert len(x) == len(self.inputs), \
            "Expected x to contain {0} inputs (received {1})".format(len(self.inputs), len(x))

        if session is None:
            session = get_session()

        feed_dict = dict(zip(self.inputs, x))
        return session.run(self.outputs, feed_dict)
