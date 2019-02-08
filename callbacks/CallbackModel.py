import tensorflow as tf
from keras.backend import get_session
import numpy as np


class CallbackModel(object):
    def __init__(self,
                 inputs: tf.Tensor,
                 outputs: tf.Tensor,
                 output_is_summary=False):
        self.inputs = inputs
        self.outputs = outputs
        self.output_is_summary = output_is_summary

    def run(self, x, session=None):
        if self.output_is_summary:
            return self.run_summary(x, session)
        else:
            return self.predict(x, session)

    def predict(self, x, batch_size=32, session=None):
        if session is None:
            session = get_session()

        batch_count = int(np.ceil(len(x) / batch_size))
        results = [None] * batch_count

        for i in range(batch_count):
            start, end = i * batch_size, (i + 1) * batch_size
            results[i] = session.run(self.outputs,
                                     feed_dict={self.inputs: x[start: end]})
        results = np.concatenate(results)

        return results

    def run_summary(self, x, session=None):
        if session is None:
            session = get_session()

        return session.run(self.outputs, feed_dict={self.inputs: x})
