from typing import Optional

from models import AutoEncoderBaseModel
from KanervaMachine import KanervaMachine


class KGDM(AutoEncoderBaseModel):
    """ KGDM : Kanerva Generative Distributed Memory """

    def __init__(self):
        super(KGDM, self).__init__()

        self.machine: Optional[KanervaMachine] = None

    def compile(self):
        pass
