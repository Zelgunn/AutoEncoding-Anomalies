from keras.callbacks import TensorBoard
import numpy as np
from typing import List

from callbacks import TensorBoardPlugin, CallbackModel


class ImageCallback(TensorBoardPlugin):
    def __init__(self,
                 summary_model: CallbackModel,
                 model_inputs: np.ndarray or List,
                 tensorboard: TensorBoard,
                 update_freq: int or str):
        super(ImageCallback, self).__init__(tensorboard, update_freq)
        self.summary_model = summary_model
        self.model_inputs = model_inputs

    def _write_logs(self, index):
        images_summaries = self.summary_model.run(self.model_inputs)
        for summaries in images_summaries:
            self.writer.add_summary(summaries, index)
