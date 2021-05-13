import tensorflow as tf
from tensorflow.python.keras import Model
from abc import abstractmethod
import numpy as np
import os
from typing import Dict, Tuple, Callable, List, Any, Union

from protocols import DatasetProtocol
from protocols.utils import make_encoder, make_decoder
from callbacks.configs import AUCCallbackConfig
from modalities import Pattern, ModalityLoadInfo, RawVideo, RawAudio, Faces, MelSpectrogram
from data_processing.video_processing import make_video_preprocessor, extract_faces
from data_processing.audio_processing import MFCCProcessor
from custom_tf_models.multimodal import ModalSync, MMAE


class AudioVideoProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 base_log_dir: str,
                 epoch: int,
                 config: Dict = None,
                 ):
        self.base_log_dir = base_log_dir
        super(AudioVideoProtocol, self).__init__(dataset_name=dataset_name,
                                                 protocol_name="audio_video",
                                                 base_log_dir=base_log_dir,
                                                 epoch=epoch,
                                                 config=config)
        self._mfcc_processor = None

    # region Make model
    def make_model(self) -> Model:
        if self.model_architecture == "modal_sync":
            model = self.make_modal_sync()
        elif self.model_architecture == "mmae":
            model = self.make_mmae()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

        self.setup_model(model)
        return model

    def setup_model(self, model: Model):
        audio_input_shape = self.to_batch_shape(self.audio_shape)
        video_input_shape = self.to_batch_shape(self.video_shape)
        model.build([audio_input_shape, video_input_shape])

        audio_input_shape = self.to_batch_shape(self.audio_shape, use_batch_size=True)
        video_input_shape = self.to_batch_shape(self.video_shape, use_batch_size=True)
        # noinspection PyProtectedMember
        model._set_inputs((tf.zeros(audio_input_shape), tf.zeros(video_input_shape)))

        model.compile(optimizer=self.make_base_optimizer())

    def make_modal_sync(self) -> ModalSync:
        audio_encoder = self.make_encoder(input_shape=self.audio_shape, flatten_code=True, name="AudioEncoder")
        video_encoder = self.make_encoder(input_shape=self.video_shape, flatten_code=True, name="VideoEncoder")

        audio_code_size = audio_encoder.compute_output_shape((None, *self.audio_shape))[1]
        video_code_size = video_encoder.compute_output_shape((None, *self.video_shape))[1]
        energy_model_input_shape = (audio_code_size + video_code_size,)

        filters = [256, 128, 64, 32]
        energy_model = make_encoder(input_shape=energy_model_input_shape,
                                    mode="dense",
                                    filters=filters,
                                    kernel_size=[1] * len(filters),
                                    strides=[1] * len(filters),
                                    code_size=1,
                                    code_activation="linear",
                                    use_code_bias=True,
                                    flatten_code=False,
                                    name="EnergyModel",
                                    )

        model = ModalSync(encoders=[audio_encoder, video_encoder],
                          energy_model=energy_model,
                          energy_margin=None)
        return model

    def make_mmae(self) -> MMAE:
        audio_encoder = self.make_encoder(input_shape=self.audio_shape, flatten_code=False, name="AudioEncoder")
        video_encoder = self.make_encoder(input_shape=self.video_shape, flatten_code=False, name="VideoEncoder")

        audio_code_shape = audio_encoder.compute_output_shape((None, *self.audio_shape))[1:]
        video_code_shape = video_encoder.compute_output_shape((None, *self.video_shape))[1:]

        audio_code_size = np.prod(audio_code_shape)
        video_code_size = np.prod(video_code_shape)
        fusion_code_size = audio_code_size + video_code_size

        filters = [fusion_code_size // 4, fusion_code_size // 16, fusion_code_size // 4]
        fusion_model = make_encoder((fusion_code_size,),
                                    mode="dense",
                                    filters=filters,
                                    kernel_size=[1] * len(filters),
                                    strides=[1] * len(filters),
                                    code_size=fusion_code_size,
                                    code_activation="relu",
                                    use_code_bias=True,
                                    flatten_code=False,
                                    name="EnergyModel",
                                    )

        audio_decoder = self.make_decoder(input_shape=audio_code_shape, name="AudioDecoder")
        video_decoder = self.make_decoder(input_shape=video_code_shape, name="VideoDecoder")

        model = MMAE(encoders=[audio_encoder, video_encoder],
                     decoders=[audio_decoder, video_decoder],
                     fusion_model=fusion_model,
                     reconstruction_loss_function=self.get_modal_sync_pretrained_error_function(),
                     multi_modal_loss=True)
        return model

    # endregion

    # region Make sub-models

    def make_encoder(self, input_shape, flatten_code=False, name="Encoder") -> Model:
        lower_case_name = name.lower()
        if "audio" in lower_case_name:
            return self.make_audio_encoder(input_shape, flatten_code, name)
        elif "video" in lower_case_name:
            return self.make_video_encoder(input_shape, flatten_code, name)
        else:
            raise RuntimeError

    def make_decoder(self, input_shape, name="Decoder") -> Model:
        lower_case_name = name.lower()
        if "audio" in lower_case_name:
            return self.make_audio_decoder(input_shape, name)
        elif "video" in lower_case_name:
            return self.make_video_decoder(input_shape, name)
        else:
            raise RuntimeError

    # region Audio (sub-models)

    def make_audio_encoder(self, input_shape, flatten_code=False, name="AudioEncoder") -> Model:
        encoder = make_encoder(input_shape=input_shape,
                               mode=self.audio_encoder_mode,
                               filters=self.audio_encoder_filters,
                               kernel_size=self.audio_encoder_kernel_sizes,
                               strides=self.audio_encoder_strides,
                               code_size=self.audio_code_size,
                               code_activation=self.code_activation,
                               use_code_bias=True,
                               basic_block_count=self.audio_encoder_basic_block_count,
                               flatten_code=flatten_code,
                               name=name,
                               )
        return encoder

    def make_audio_decoder(self, input_shape, name="AudioDecoder") -> Model:
        encoder = make_decoder(input_shape=input_shape,
                               mode=self.audio_encoder_mode,
                               filters=self.audio_encoder_filters,
                               kernel_size=self.audio_encoder_kernel_sizes,
                               stem_kernel_size=self.audio_decoder_stem_size,
                               strides=self.audio_encoder_strides,
                               channels=self.audio_channels,
                               output_activation=self.output_activation,
                               basic_block_count=self.audio_encoder_basic_block_count,
                               name=name,
                               )
        return encoder

    # endregion

    # region Video (sub-models)

    def make_video_encoder(self, input_shape, flatten_code=False, name="VideoEncoder") -> Model:
        encoder = make_encoder(input_shape=input_shape,
                               mode=self.video_encoder_mode,
                               filters=self.video_encoder_filters,
                               kernel_size=self.video_encoder_kernel_sizes,
                               strides=self.video_encoder_strides,
                               code_size=self.video_code_size,
                               code_activation=self.code_activation,
                               use_code_bias=True,
                               basic_block_count=self.video_encoder_basic_block_count,
                               flatten_code=flatten_code,
                               name=name,
                               )
        return encoder

    def make_video_decoder(self, input_shape, name="VideoDecoder") -> Model:
        encoder = make_decoder(input_shape=input_shape,
                               mode=self.video_encoder_mode,
                               filters=self.video_encoder_filters,
                               kernel_size=self.video_encoder_kernel_sizes,
                               stem_kernel_size=self.video_decoder_stem_size,
                               strides=self.video_encoder_strides,
                               channels=self.video_channels,
                               output_activation=self.output_activation,
                               basic_block_count=self.video_encoder_basic_block_count,
                               name=name,
                               )
        return encoder

    # endregion

    # endregion

    # region Patterns
    def get_train_pattern(self) -> Pattern:
        augmented_preprocessor = self.make_augmented_preprocessor()

        video_infos = self.get_video_modality_info(training=True)
        audio_infos = self.get_audio_modality_info(training=True)
        modalities_infos = [audio_infos, *video_infos]

        pattern = Pattern(
            *modalities_infos,
            preprocessor=augmented_preprocessor
        )

        return pattern

    def get_anomaly_pattern(self) -> Pattern:
        preprocessor = self.make_preprocessor(include_labels=True)

        audio_infos = self.get_audio_modality_info(training=False)
        video_infos = self.get_video_modality_info(training=False)
        modalities_infos = [audio_infos, *video_infos]

        pattern = Pattern(
            *modalities_infos,
            preprocessor=preprocessor
        )

        return pattern.with_labels()

    def get_audio_modality_info(self, training: bool) -> ModalityLoadInfo:
        audio_length = self.get_audio_length(training)

        if self.use_mfcc:
            audio_infos = ModalityLoadInfo(MelSpectrogram, length=audio_length)
        else:
            audio_infos = ModalityLoadInfo(RawAudio, length=audio_length)

        return audio_infos

    def get_video_modality_info(self, training: bool) -> List[ModalityLoadInfo]:
        video_length = self.get_video_length(training)

        video_infos = [ModalityLoadInfo(RawVideo, length=video_length)]
        if self.use_face:
            video_infos += [ModalityLoadInfo(Faces, length=video_length)]

        return video_infos

    # endregion

    # region Data preprocess/augmentation
    def make_preprocessor(self, include_labels: bool) -> Callable:
        video_preprocessor = make_video_preprocessor(to_grayscale=self.use_grayscale,
                                                     activation_range=self.output_activation,
                                                     include_labels=False,
                                                     target_size=self.image_size)

        def _preprocess(audio: tf.Tensor, video: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            if self.use_mfcc:
                audio = self.mfcc_processor.pre_process(audio)
            video = video_preprocessor(video)
            return audio, video

        if self.use_face:
            if include_labels:
                def preprocess(audio, video, faces, labels):
                    video = extract_faces(video, faces)
                    audio, video = _preprocess(audio, video)
                    return audio, video, labels
            else:
                def preprocess(audio, video, faces):
                    video = extract_faces(video, faces)
                    audio, video = _preprocess(audio, video)
                    return audio, video
        else:
            if include_labels:
                def preprocess(audio, video, labels):
                    audio, video = _preprocess(audio, video)
                    return audio, video, labels
            else:
                def preprocess(audio, video):
                    return _preprocess(audio, video)

        return preprocess

    def make_augmented_preprocessor(self, include_labels=False) -> Callable:
        base_preprocessor = self.make_preprocessor(include_labels)

        def _augment_data(audio: tf.Tensor, video: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            return audio, video  # TODO : Add actual augmentation

        if self.use_face:
            if include_labels:
                def augmented_preprocessor(audio, video, faces, labels):
                    audio, video, labels = base_preprocessor(audio, video, faces, labels)
                    audio, video = _augment_data(audio, video)
                    return audio, video
            else:
                def augmented_preprocessor(audio, video, faces):
                    audio, video = base_preprocessor(audio, video, faces)
                    audio, video = _augment_data(audio, video)
                    return audio, video
        else:
            if include_labels:
                def augmented_preprocessor(audio, video, labels):
                    audio, video, labels = base_preprocessor(audio, video, labels)
                    audio, video = _augment_data(audio, video)
                    return audio, video
            else:
                def augmented_preprocessor(audio, video):
                    audio, video = base_preprocessor(audio, video)
                    audio, video = _augment_data(audio, video)
                    return audio, video

        return augmented_preprocessor

    @property
    def mfcc_processor(self):
        if self._mfcc_processor is None:
            self._mfcc_processor = MFCCProcessor(activation_range=self.output_activation)
        return self._mfcc_processor

    # endregion

    # region Pre-trained modules
    def get_pretrained_modal_sync_model(self) -> ModalSync:
        load_path = os.path.join(self.base_log_dir, "audio_video/audioset/pretrained_modal_sync_73")
        modal_sync_model = tf.keras.models.load_model(load_path)
        return modal_sync_model

    def get_modal_sync_pretrained_error_function(self) -> Callable[[List[tf.Tensor], List[tf.Tensor]], tf.Tensor]:
        modal_sync_model = self.get_pretrained_modal_sync_model()
        modal_sync_model.trainable = False

        def error_function(inputs: List[tf.Tensor], outputs: List[tf.Tensor]) -> tf.Tensor:
            inputs_encoded = modal_sync_model.encode(inputs)
            outputs_encoded = modal_sync_model.encode(outputs)
            return tf.abs(inputs_encoded - outputs_encoded)

        return error_function

    # endregion

    # region Callbacks

    def get_auc_callback_configs(self) -> List[AUCCallbackConfig]:
        if self.auc_frequency < 1:
            return []

        anomaly_pattern = self.get_anomaly_pattern()
        auc_callbacks_configs = []

        model = self.model

        if isinstance(model, MMAE):
            auc_callbacks_configs += [
                AUCCallbackConfig(model, anomaly_pattern, labels_length=1, prefix="AE",
                                  convert_to_io_compare_model=True, epoch_freq=self.auc_frequency,
                                  io_compare_metrics="multi_modal_mae", sample_count=self.auc_sample_count),
            ]

        return auc_callbacks_configs

    # endregion

    # region Properties
    @property
    def encoder_input_shape(self) -> Tuple[int, ...]:
        raise RuntimeError("Not a single encoder input shape can be specified for audio-video protocols.")

    @property
    def input_time(self) -> float:
        return self.config["input_time"]

    # region Audio properties
    # region Audio config
    @property
    def audio_config(self) -> Dict[str, Any]:
        return self.config["audio"]

    def get_audio_config_value(self, key: str, default: Any) -> Any:
        if key not in self.audio_config:
            self.audio_config[key] = default
        return self.audio_config[key]

    # endregion

    # region Audio shape
    @property
    def audio_length(self) -> int:
        return int(self.input_time * self.audio_sample_rate)

    @property
    def audio_train_length(self) -> int:
        if self.model_architecture in ["modal_sync"]:
            desync_time = self.get_config_value("desync_time", default=2.0)
            additional_length = int(desync_time * self.audio_sample_rate)
            return self.audio_length + additional_length
        else:
            return self.audio_length

    def get_audio_length(self, training: bool) -> int:
        return self.audio_train_length if training else self.audio_length

    @property
    def audio_channels(self):
        if self.use_mfcc:
            if "channels" not in self.audio_config:
                raise RuntimeError("When using MelSpectrogram, `channels` must be specified in the audio config.")
            return self.audio_config["channels"]
        else:
            return self.get_audio_config_value("channels", default=1)

    @property
    def audio_shape(self) -> Tuple[int, int]:
        return self.audio_length, self.audio_channels

    # endregion

    @property
    def use_mfcc(self) -> bool:
        return self.get_audio_config_value("use_mfcc", default=True)

    @property
    @abstractmethod
    def audio_sample_rate(self) -> int:
        raise NotImplementedError("Audio sample rate must be specified in sub-classes.")

    # region Audio encoder properties
    @property
    def audio_encoder_config(self) -> Dict[str, Any]:
        return self.audio_config["encoder"]

    @property
    def audio_encoder_mode(self) -> str:
        return self.audio_encoder_config["mode"]

    @property
    def audio_encoder_filters(self) -> List[int]:
        return self.audio_encoder_config["filters"]

    @property
    def audio_encoder_depth(self) -> int:
        return len(self.audio_encoder_filters)

    @property
    def audio_encoder_kernel_sizes(self) -> Union[int, List[int]]:
        kernel_sizes = self.audio_encoder_config["kernel_sizes"]
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * self.audio_encoder_depth
        return kernel_sizes

    @property
    def audio_encoder_strides(self) -> Union[int, List[int]]:
        strides = self.audio_encoder_config["strides"]
        if not isinstance(strides, (list, tuple)):
            strides = [strides] * self.audio_encoder_depth
        return strides

    @property
    def audio_code_size(self) -> int:
        return self.audio_encoder_config["code_size"]

    @property
    def audio_encoder_basic_block_count(self) -> int:
        return self.audio_encoder_config["basic_block_count"]

    # endregion

    # region Audio decoder properties
    @property
    def audio_decoder_config(self) -> Dict[str, Any]:
        return self.audio_config["decoder"]

    @property
    def audio_decoder_mode(self) -> str:
        return self.audio_decoder_config["mode"]

    @property
    def audio_decoder_filters(self) -> List[int]:
        return self.audio_decoder_config["filters"]

    @property
    def audio_decoder_depth(self) -> int:
        return len(self.audio_decoder_filters)

    @property
    def audio_decoder_kernel_sizes(self) -> Union[int, List[int]]:
        kernel_sizes = self.audio_decoder_config["kernel_sizes"]
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * self.audio_decoder_depth
        return kernel_sizes

    @property
    def audio_decoder_strides(self) -> Union[int, List[int]]:
        strides = self.audio_decoder_config["strides"]
        if not isinstance(strides, (list, tuple)):
            strides = [strides] * self.audio_decoder_depth
        return strides

    @property
    def audio_decoder_stem_size(self) -> int:
        return self.audio_decoder_config["stem_size"]

    @property
    def audio_decoder_basic_block_count(self) -> int:
        return self.audio_decoder_config["basic_block_count"]

    # endregion

    # endregion

    # region Video properties
    # region Video config
    @property
    def video_config(self) -> Dict[str, Any]:
        return self.config["video"]

    def get_video_config_value(self, key: str, default: Any) -> Any:
        if key not in self.video_config:
            self.video_config[key] = default
        return self.video_config[key]

    # endregion

    # region Video shape
    @property
    def video_length(self) -> int:
        return int(self.input_time * self.video_sample_rate)

    @property
    def video_train_length(self) -> int:
        if self.model_architecture in ["modal_sync"]:
            desync_time = self.get_config_value("desync_time", default=2.0)
            additional_length = int(desync_time * self.video_sample_rate)
            return self.video_length + additional_length
        else:
            return self.video_length

    def get_video_length(self, training: bool) -> int:
        return self.video_train_length if training else self.video_length

    @property
    def image_size(self) -> Tuple[int, int]:
        return tuple(self.video_config["image_size"])

    @property
    def video_height(self) -> int:
        return self.image_size[0]

    @property
    def video_width(self) -> int:
        return self.image_size[1]

    @property
    def video_channels(self) -> int:
        return self.get_video_config_value("channels", default=1)

    @property
    def video_shape(self) -> Tuple[int, int, int, int]:
        return self.video_length, self.video_width, self.video_height, self.video_channels

    @property
    def use_grayscale(self) -> bool:
        return self.video_channels == 1

    # endregion

    @property
    def use_face(self) -> bool:
        return self.get_video_config_value("use_face", default=False)

    @property
    @abstractmethod
    def video_sample_rate(self) -> int:
        raise NotImplementedError("Video sample rate must be specified in sub-classes.")

    # region Video encoder properties
    @property
    def video_encoder_config(self) -> Dict[str, Any]:
        return self.video_config["encoder"]

    @property
    def video_encoder_mode(self) -> str:
        return self.video_encoder_config["mode"]

    @property
    def video_encoder_filters(self) -> List[int]:
        return self.video_encoder_config["filters"]

    @property
    def video_encoder_depth(self) -> int:
        return len(self.video_encoder_filters)

    @property
    def video_encoder_kernel_sizes(self) -> Union[int, List[int]]:
        kernel_sizes = self.video_encoder_config["kernel_sizes"]

        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * self.video_encoder_depth

        return kernel_sizes

    @property
    def video_encoder_strides(self) -> Union[int, List[int]]:
        strides = self.video_encoder_config["strides"]

        if not isinstance(strides, list):
            strides = [strides] * self.video_encoder_depth

        return strides

    @property
    def video_code_size(self) -> int:
        return self.video_encoder_config["code_size"]

    @property
    def video_encoder_basic_block_count(self) -> int:
        return self.video_encoder_config["basic_block_count"]

    # endregion

    # region Video decoder properties
    @property
    def video_decoder_config(self) -> Dict[str, Any]:
        return self.video_config["decoder"]

    @property
    def video_decoder_mode(self) -> str:
        return self.video_decoder_config["mode"]

    @property
    def video_decoder_filters(self) -> List[int]:
        return self.video_decoder_config["filters"]

    @property
    def video_decoder_depth(self) -> int:
        return len(self.video_decoder_filters)

    @property
    def video_decoder_kernel_sizes(self) -> Union[int, List[int]]:
        kernel_sizes = self.video_decoder_config["kernel_sizes"]
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * self.video_decoder_depth
        return kernel_sizes

    @property
    def video_decoder_strides(self) -> Union[int, List[int]]:
        strides = self.video_decoder_config["strides"]
        if not isinstance(strides, (list, tuple)):
            strides = [strides] * self.video_decoder_depth
        return strides

    @property
    def video_decoder_stem_size(self) -> int:
        return self.video_decoder_config["stem_size"]

    @property
    def video_decoder_basic_block_count(self) -> int:
        return self.video_decoder_config["basic_block_count"]

    # endregion

    # endregion
    # endregion
