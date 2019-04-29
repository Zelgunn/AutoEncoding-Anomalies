import tensorflow as tf
from typing import Optional

from models import AutoEncoderBaseModel
from KanervaMemory import Memory


class KGDM(AutoEncoderBaseModel):
    """ KGDM : Kanerva Generative Distributed Memory """

    def __init__(self):
        super(KGDM, self).__init__()

        self.memory: Optional[Memory] = None

    def compile(self):
        pass
