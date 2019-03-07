from keras .models import Model as KerasModel
from keras.callbacks import Callback
from typing import Dict


class MultipleModelsCheckpoint(Callback):
    def __init__(self,
                 base_filepath: str,
                 models: Dict[str, KerasModel],
                 period: int):
        super(MultipleModelsCheckpoint, self).__init__()
        self.models = models
        self.period = period
        if "{model}" not in base_filepath:
            base_filepath = base_filepath + "_{model}"
        self.base_filepath = base_filepath

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % self.period) == 0:
            for model_name, model in self.models.items():
                filepath = self.base_filepath.format(model=model_name, epoch=epoch + 1, **logs)
                model.save_weights(filepath, overwrite=True)
