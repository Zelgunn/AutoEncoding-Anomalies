import tensorflow as tf
from tensorflow.python.keras.backend import get_session
from tensorflow.python.keras.utils.generic_utils import to_list
import numpy as np
from typing import List


class RunModel(object):
    def __init__(self,
                 inputs: tf.Tensor or List,
                 outputs: tf.Tensor or List,
                 output_is_summary=False):
        self.inputs = to_list(inputs)
        self.outputs = to_list(outputs)
        self.output_is_summary = output_is_summary

    def run(self, x, session=None):
        if self.output_is_summary:
            return self.run_summary(x, session)
        else:
            return self.predict(x, session)

    def predict(self, x, batch_size=32, session=None):
        x = to_list(x)
        assert len(x) == len(self.inputs),\
            "Expected x to contain {0} inputs (received {1})".format(len(self.inputs), len(x))

        if session is None:
            session = get_session()

        total_size = len(x[0])
        batch_count = int(np.ceil(total_size / batch_size))

        batches_results = [None] * batch_count

        for i in range(batch_count):
            start, end = i * batch_size, (i + 1) * batch_size
            values = [array[start:end] for array in x]
            feed_dict = dict(zip(self.inputs, values))
            batches_results[i] = session.run(self.outputs, feed_dict)

        results = []
        for i in range(len(self.outputs)):
            ith_output_results = [batch_results[i] for batch_results in batches_results]
            ith_output_results = np.concatenate(ith_output_results, axis=0)
            results.append(ith_output_results)

        return results

    def run_summary(self, x, session=None):
        x = to_list(x)
        assert len(x) == len(self.inputs), \
            "Expected x to contain {0} inputs (received {1})".format(len(self.inputs), len(x))

        if session is None:
            session = get_session()

        feed_dict = dict(zip(self.inputs, x))
        return session.run(self.outputs, feed_dict)
