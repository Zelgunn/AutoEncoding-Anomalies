from models import VariationalBaseModel, KerasModel
from models.VariationalBaseModel import kullback_leibler_divergence_mean0_var1


class VAE(VariationalBaseModel):
    def compile(self):
        input_layer = self.encoder.input
        encoded, latent_mean, latent_log_var = self.encoder(input_layer)
        autoencoded = self.decoder(encoded)

        def vae_loss(y_true, y_pred):
            reconstruction_loss_function = self.get_reconstruction_loss(self.config["reconstruction_loss"])
            reconstruction_loss = reconstruction_loss_function(y_true, y_pred)
            divergence = kullback_leibler_divergence_mean0_var1(latent_mean, latent_log_var)
            return reconstruction_loss + divergence * 0.001

        autoencoder = KerasModel(inputs=input_layer, outputs=autoencoded, name="AutoEncoder")
        autoencoder.compile(self.optimizer, loss=vae_loss, metrics=self.config["metrics"])

        self._autoencoder = autoencoder
        self._latent_mean = latent_mean
        self._latent_log_var = latent_log_var


