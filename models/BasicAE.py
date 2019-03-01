from models import AutoEncoderBaseModel, KerasModel, metrics_dict


class BasicAE(AutoEncoderBaseModel):
    def compile(self):
        autoencoded = self.decoder(self.encoder(self.encoder.input))
        autoencoder = KerasModel(inputs=self.encoder.inputs, outputs=autoencoded)
        reconstruction_loss = metrics_dict[self.config["reconstruction_loss"]]
        autoencoder.compile(self.optimizer, reconstruction_loss, metrics=self.config["metrics"])

        self._autoencoder = autoencoder
