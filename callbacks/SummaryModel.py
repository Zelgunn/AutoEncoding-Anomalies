import tensorflow as tf
from keras.backend import get_session


class SummaryModel(object):
    def __init__(self,
                 inputs: tf.Tensor,
                 outputs: tf.Tensor):
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, x, session=None):
        if session is None:
            session = get_session()

        return session.run(self.outputs, feed_dict={self.inputs: x})
