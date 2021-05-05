from tensorflow.python.keras import Model
from typing import Dict, Tuple, Callable

from protocols import DatasetProtocol
from modalities import Pattern, ModalityLoadInfo, RawVideo, RawAudio
from data_processing.video_processing.video_preprocessing import make_video_augmentation
from custom_tf_models.multimodal import ModalSync


class AudioVideoProtocol(DatasetProtocol):
    def __init__(self,
                 dataset_name: str,
                 base_log_dir: str,
                 epoch: int,
                 config: Dict = None,
                 ):
        super(AudioVideoProtocol, self).__init__(dataset_name=dataset_name,
                                                 protocol_name="audio_video",
                                                 base_log_dir=base_log_dir,
                                                 epoch=epoch,
                                                 config=config)

    # region Make model
    def make_model(self) -> Model:
        if self.model_architecture == "modal_sync":
            model = self.make_modal_sync()
        else:
            raise ValueError("Unknown architecture : {}.".format(self.model_architecture))

        self.setup_model(model)
        return model

    def make_modal_sync(self) -> ModalSync:
        audio_encoder = self.make_encoder(input_shape=self.audio_input_shape, name="AudioEncoder")
        video_encoder = self.make_encoder(input_shape=self.video_input_shape, name="VideoEncoder")
        energy_model = self.make_encoder(input_shape=self.energy_model_input_shape, name="EnergyModel")

        model = ModalSync(encoders=[audio_encoder, video_encoder],
                          energy_model=energy_model,
                          energy_margin=None)
        return model

    # endregion

    # region Patterns
    def get_train_pattern(self) -> Pattern:
        # augment_video = self.make_video_augmentation()

        pattern = Pattern(
            ModalityLoadInfo(RawVideo, self.output_length),
            ModalityLoadInfo(RawAudio, self.output_length),
            # preprocessor=augment_video
        )

        return Pattern

    def get_anomaly_pattern(self) -> Pattern:
        raise NotImplementedError

    def get_audio_modality_info(self):
        if self.use_mfcc:
            pass

    def get_video_modality_info(self):
        if self.use_face:
            pass

    # endregion

    # region Data preprocess/augmentation
    def make_video_augmentation(self) -> Callable:
        return make_video_augmentation(length=self.video_train_length,
                                       height=self.height,
                                       width=self.width,
                                       channels=self.dataset_channels,
                                       dropout_noise_ratio=self.dropout_noise_ratio,
                                       negative_prob=negative_prob,
                                       activation_range=self.output_activation)

    # endregion

    @property
    def encoder_input_shape(self) -> Tuple[int, ...]:
        raise RuntimeError

    # region Audio properties
    @property
    def audio_length(self) -> int:
        return self.config["audio_length"]

    @property
    def audio_train_length(self) -> int:
        if self.model_architecture in ["modal_sync"]:
            desync_time = self.get_config_value("desync_time", default=2.0)
            additional_length = int(desync_time * self.audio_sample_rate)
            return self.audio_length + additional_length
        else:
            return self.audio_length

    @property
    def audio_channels(self):
        return self.get_config_value("audio_channels", default=1)

    @property
    def use_mfcc(self) -> bool:
        return self.get_config_value("use_mfcc", default=True)

    @property
    def audio_sample_rate(self) -> int:
        raise NotImplementedError("Audio sample rate must be specified in sub-classes.")

    # endregion

    # region Video properties
    @property
    def video_length(self) -> int:
        return self.config["video_length"]

    @property
    def video_train_length(self) -> int:
        if self.model_architecture in ["modal_sync"]:
            desync_time = self.get_config_value("desync_time", default=2.0)
            additional_length = int(desync_time * self.video_sample_rate)
            return self.video_length + additional_length
        else:
            return self.video_length

    @property
    def image_shape(self) -> Tuple[int, int]:
        return self.config["image_shape"]

    @property
    def video_height(self) -> int:
        return self.image_shape[0]

    @property
    def video_width(self) -> int:
        return self.image_shape[1]

    @property
    def video_channels(self):
        return self.get_config_value("video_channels", default=1)

    @property
    def use_face(self) -> bool:
        return True

    @property
    def video_sample_rate(self) -> int:
        raise NotImplementedError("Video sample rate must be specified in sub-classes.")

    # endregion
