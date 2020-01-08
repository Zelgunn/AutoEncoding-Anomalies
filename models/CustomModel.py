import tensorflow as tf
from tensorflow.python.keras import Model, regularizers
from tensorflow.python.keras.callbacks import Callback, CallbackList, configure_callbacks, make_logs
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.data.ops.dataset_ops import get_legacy_output_shapes
from abc import abstractmethod
from typing import List, Dict, Type, Optional
import os

from misc_utils.train_utils import LossAggregator
from misc_utils.summary_utils import tf_function_summary


class CustomModel(Model):
    # region Training
    def fit(self,
            x: tf.data.Dataset = None,
            batch_size: int = None,
            epochs=1,
            callbacks: List[Callback] = None,
            validation_data: tf.data.Dataset = None,
            initial_epoch=0,
            steps_per_epoch: int = None,
            validation_steps: int = None,
            validation_freq=1,
            verbose=1,
            **kwargs):

        do_validation = (validation_data is not None) and (validation_steps is not None)
        model_checkpoint = get_callback(callbacks, ModelCheckpoint)
        tensorboard: Optional[TensorBoard] = get_callback(callbacks, TensorBoard)
        callbacks: CallbackList = configure_callbacks(callbacks,
                                                      model=self,
                                                      do_validation=do_validation,
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      steps_per_epoch=steps_per_epoch,
                                                      samples=steps_per_epoch,
                                                      verbose=verbose,
                                                      mode=ModeKeys.TRAIN)
        if model_checkpoint is not None:
            model_checkpoint.save_weights_only = True

        if (tensorboard is not None) and (tensorboard.update_freq != "epoch"):
            tensorboard._samples_seen = initial_epoch * steps_per_epoch
            tensorboard._total_batches_seen = initial_epoch * steps_per_epoch

        # self.write_model_graph(tensorboard, dataset=x)

        train_aggregator = LossAggregator(use_steps=True, num_samples=steps_per_epoch)
        val_aggregator = LossAggregator(use_steps=True, num_samples=validation_steps)

        callbacks.on_train_begin()
        for epoch in range(initial_epoch, epochs):
            if callbacks.model.stop_training:
                break

            epoch_logs = {}
            callbacks.on_epoch_begin(epoch, epoch_logs)

            # region Training
            for step, batch in zip(range(steps_per_epoch), x):
                batch_logs = {'batch': step, 'size': 1}
                callbacks.on_batch_begin(step, batch_logs)

                # low_states = self.get_low_energy_states(batch)
                # high_states = self.get_high_energy_states(batch)
                # states = low_states + high_states
                #
                # audio_samples, video_samples = [], []
                # for state in states:
                #     audio, video = state[0]
                #     audio_samples.append(audio.numpy())
                #     video_samples.append(video.numpy())
                #
                # import numpy as np
                # audio_samples = np.concatenate(audio_samples, axis=0)
                # video_samples = np.concatenate(video_samples, axis=0)
                # video_samples = np.tile(video_samples, [1, 1, 1, 1, 3])
                # # audio_samples = audio_samples[:2]
                # # video_samples = video_samples[:2]
                #
                # print(video_samples.max(), video_samples.min())
                # print(audio_samples.max(), audio_samples.min())
                # exit()

                # from modalities import MelSpectrogram
                # ms = MelSpectrogram(0.03, 0.01005, 100, to_db=True)
                # audio_samples = (audio_samples - 1.0) * 80.0
                # audio_samples = ms.mel_spectrograms_to_wave(audio_samples, 48000)
                # audio_samples = np.expand_dims(audio_samples, axis=-1)
                # video_samples = (video_samples * 255.0).astype(np.uint8)
                #
                # from modalities.utils import write_video_with_audio
                # for i in range(len(video_samples)):
                #     path = r"..\logs\AEA\protocols\audio_video\emoly\train\{}.avi".format(i)
                #     write_video_with_audio(path, video_samples[i], audio_samples[i], 25, 48000)
                # exit()

                batch_outputs = self.train_step(batch)
                if not (isinstance(batch_outputs, tuple) or isinstance(batch_outputs, list)):
                    batch_outputs = [batch_outputs]
                batch_outputs = [output.numpy() for output in batch_outputs]

                if step == 0:
                    train_aggregator.create(batch_outputs)
                train_aggregator.aggregate(batch_outputs)

                batch_logs = make_logs(self, batch_logs, batch_outputs, ModeKeys.TRAIN)

                callbacks.on_batch_end(step, batch_logs)

                if callbacks.model.stop_training:
                    break

            train_aggregator.finalize()
            epoch_logs = make_logs(self, epoch_logs, train_aggregator.results, ModeKeys.TRAIN)
            # endregion

            # region Validation
            if do_validation and (epoch % validation_freq) == 0:
                for val_step, batch in zip(range(validation_steps), validation_data):
                    val_results = self.compute_metrics(batch)
                    if not (isinstance(val_results, tuple) or isinstance(val_results, list)):
                        val_results = [val_results]
                    val_results = [output.numpy() for output in val_results]

                    if val_step == 0:
                        val_aggregator.create(val_results)
                    val_aggregator.aggregate(val_results)

                val_aggregator.finalize()
                epoch_logs = make_logs(self, epoch_logs, val_aggregator.results, ModeKeys.TRAIN, prefix="val_")
            # endregion

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()
        return self.history

    @abstractmethod
    def train_step(self, inputs, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, inputs, *args, **kwargs):
        pass

    def compute_weights_decay_loss(self, l1=0.0, l2=0.0):
        loss = 0
        for variable in self.trainable_variables:
            loss += regularizers.L1L2(l1=l1, l2=l2)(variable)
        return loss

    def compute_metrics(self, inputs, *args, **kwargs):
        return self.compute_loss(inputs, *args, **kwargs)

    @tf.function
    def forward(self, inputs):
        return self(inputs)

    # endregion

    def compute_output_signature(self, input_signature):
        pass

    # region Summary/Save/Load
    @property
    @abstractmethod
    def models_ids(self) -> Dict[Model, str]:
        pass

    @abstractmethod
    def get_config(self):
        pass

    def summary(self, line_length=None, positions=None, print_fn=None):
        for model in self.models_ids.keys():
            model.summary(line_length=line_length, positions=positions, print_fn=print_fn)

    def write_model_graph(self, tensorboard: TensorBoard, dataset: tf.data.Dataset):
        shapes = [get_legacy_output_shapes(dataset)]

        # noinspection PyProtectedMember
        with tensorboard._get_writer(tensorboard._train_run_name).as_default():
            tf_function_summary(self.forward, shapes, name="train_step")

    def save(self,
             filepath: str,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None
             ):
        last_point_index = filepath.rindex(".")
        filepath = filepath[:last_point_index] + "_{}" + filepath[last_point_index:]
        for model, model_id in self.models_ids.items():
            model.save(filepath.format(model_id), overwrite=overwrite,
                       include_optimizer=include_optimizer, save_format=save_format,
                       signatures=signatures, options=options)

    def save_weights(self, filepath, overwrite=True, save_format=None):
        last_point_index = filepath.rindex(".")
        filepath = filepath[:last_point_index] + "_{}" + filepath[last_point_index:]
        for model, model_id in self.models_ids.items():
            model.save_weights(filepath.format(model_id), overwrite=overwrite, save_format=save_format)

    def load_weights(self,
                     filepath: str,
                     by_name=False):
        # tmp = filepath
        last_point_index = filepath.rindex(".")
        base_filepath = filepath[:last_point_index] + "_{}" + filepath[last_point_index:]
        result = None
        for model, model_id in self.models_ids.items():
            if isinstance(model, CustomModel):
                result = model.load_weights(base_filepath.format(model_id), by_name=by_name)
            else:
                model_filepath = base_filepath.format(model_id)
                if os.path.isfile(model_filepath):
                    result = model.load_weights(model_filepath, by_name=by_name)
                    print("Successfully loaded {} from {}.".format(model_id, model_filepath))
                else:
                    print("Could not load {} - {} is not a valid filepath.".format(model_id, model_filepath))
        # super(CustomModel, self).load_weights(tmp, by_name=by_name)
        return result
    # endregion


def get_callback(callbacks: List[Callback], callback_type: Type[Callback]) -> Optional[Callback]:
    if callbacks is None:
        return None

    result = None
    for callback in callbacks:
        if isinstance(callback, callback_type):
            result = callback
            break
    return result
