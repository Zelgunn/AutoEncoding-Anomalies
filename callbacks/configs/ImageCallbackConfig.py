from tensorflow.python.keras.callbacks import TensorBoard

from callbacks import ImageCallback
from callbacks.configs import ModalityCallbackConfig
from datasets import DatasetLoader


class ImageCallbackConfig(ModalityCallbackConfig):
    def to_callback(self,
                    tensorboard: TensorBoard,
                    dataset_loader: DatasetLoader,
                    seed=None,
                    ) -> ImageCallback:
        image_callbacks = ImageCallback.from_model_and_subset(autoencoder=self.autoencoder,
                                                              subset=self.get_subset(dataset_loader),
                                                              pattern=self.pattern,
                                                              name=self.name,
                                                              is_train_callback=self.is_train_callback,
                                                              tensorboard=tensorboard,
                                                              epoch_freq=self.epoch_freq,
                                                              inputs_are_outputs=self.inputs_are_outputs,
                                                              modality_index=self.modality_indices,
                                                              seed=seed,
                                                              **self.kwargs)
        return image_callbacks
