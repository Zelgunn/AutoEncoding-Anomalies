import tensorflow as tf
from tensorflow.python.keras import Model

from models import IAE


class IAEGAN(IAE):
    def __init__(self,
                 encoder: Model,
                 decoder: Model,
                 discriminator: Model,
                 step_size: int,
                 ):
        super(IAEGAN, self).__init__(encoder=encoder,
                                     decoder=decoder,
                                     step_size=step_size)
        self.discriminator = discriminator

    def compute_loss(self,
                     inputs
                     ) -> tf.Tensor:
        pass
