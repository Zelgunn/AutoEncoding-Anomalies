from keras.callbacks import TensorBoard
import numpy as np
from typing import List

from callbacks import TensorBoardPlugin, RunModel


class ImageCallback(TensorBoardPlugin):
    def __init__(self,
                 one_shot_summary_model: RunModel,
                 repeated_summary_model: RunModel,
                 model_inputs: np.ndarray or List,
                 tensorboard: TensorBoard,
                 update_freq: int or str,
                 epoch_freq: int = None):
        super(ImageCallback, self).__init__(tensorboard, update_freq, epoch_freq)
        self.one_shot_summary_model = one_shot_summary_model
        self.repeated_summary_model = repeated_summary_model
        self.model_inputs = model_inputs

    def _write_logs(self, index):
        if self.one_shot_summary_model is not None:
            images_summaries = self.one_shot_summary_model.run(self.model_inputs)
            for summaries in images_summaries:
                self.writer.add_summary(summaries, index)

            self.one_shot_summary_model = None

        if self.repeated_summary_model is not None:
            images_summaries = self.repeated_summary_model.run(self.model_inputs)
            for summaries in images_summaries:
                self.writer.add_summary(summaries, index)
