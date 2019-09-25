from tensorflow.python.keras import Model
from typing import Dict

from protocols import VideoProtocol, ProtocolTrainConfig, ProtocolTestConfig
from protocols.utils import make_residual_encoder, make_residual_decoder
from models import IAE


class AvenueProtocol(VideoProtocol):
    def __init__(self,
                 initial_epoch=0,
                 model_name=None
                 ):
        self.initial_epoch = initial_epoch

        self.step_size = 8
        self.step_count = 4

        self.code_size = 64
        self.code_activation = "sigmoid"
        self.encoder_filters = [8, 12, 18, 27, 40]
        self.encoder_strides = [(1, 2, 2),
                                (2, 1, 1),
                                (1, 2, 2),
                                (2, 1, 1),
                                (1, 2, 2)]

        self.decoder_filters = [64, 40, 27, 18, 12, 8]
        self.decoder_strides = [(1, 1, 1),
                                (1, 2, 2),
                                (1, 2, 2),
                                (2, 1, 1),
                                (1, 2, 2),
                                (2, 1, 1)]
        self.output_activation = "linear"

        super(AvenueProtocol, self).__init__(dataset_name="avenue",
                                             protocol_name="video",
                                             height=128,
                                             width=128,
                                             model_name=model_name)

    def make_model(self) -> Model:
        input_shape = (None, self.step_size, self.height, self.width, 1)

        encoder = self.make_encoder()
        encoder.build(input_shape=input_shape)

        decoder = self.make_decoder()
        decoder.build(input_shape=encoder.compute_output_shape(input_shape))

        model = IAE(encoder=encoder,
                    decoder=decoder,
                    step_size=self.step_size)
        return model

    def make_encoder(self):
        encoder = make_residual_encoder(filters=self.encoder_filters,
                                        strides=self.encoder_strides,
                                        code_size=self.code_size,
                                        code_activation=self.code_activation)
        return encoder

    def make_decoder(self):
        decoder = make_residual_decoder(filters=self.decoder_filters,
                                        strides=self.decoder_strides,
                                        channels=1,
                                        output_activation=self.output_activation)
        return decoder

    def get_train_config(self) -> ProtocolTrainConfig:
        train_pattern = self.get_train_pattern()
        image_callbacks_configs = self.get_image_callback_configs()
        auc_callbacks_configs = self.get_auc_callbacks_configs()

        return ProtocolTrainConfig(batch_size=16,
                                   pattern=train_pattern,
                                   epochs=50,
                                   initial_epoch=self.initial_epoch,
                                   image_callbacks_configs=image_callbacks_configs,
                                   auc_callbacks_configs=auc_callbacks_configs,
                                   early_stopping_metric=self.model.metrics_names[0])

    def get_test_config(self) -> ProtocolTestConfig:
        anomaly_pattern = self.get_anomaly_pattern()
        return ProtocolTestConfig(pattern=anomaly_pattern,
                                  epoch=self.initial_epoch,
                                  output_length=self.output_length,
                                  detector_stride=1,
                                  pre_normalize_predictions=True)

    @property
    def output_length(self) -> int:
        return self.step_size * self.step_count

    def get_config(self) -> Dict:
        config = {
            "step_size": self.step_size,
            "step_count": self.step_count,
            "code_size": self.code_size,
            "height": self.height,
            "width": self.width,
            "code_activation": self.code_activation,
            "encoder_filters": self.encoder_filters,
            "encoder_strides": self.encoder_strides,
            "decoder_filters": self.decoder_filters,
            "decoder_strides": self.decoder_strides,
            "output_activation": self.output_activation,
        }
        base_config = super(AvenueProtocol, self).get_config()
        return {**base_config, **config}
