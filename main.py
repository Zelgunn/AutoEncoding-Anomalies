import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models import BasicAE, ProgAE, AGE, VAE, GAN
from datasets import UCSDDatabase

model_used = "GAN"

if model_used == "BasicAE":
    auto_encoder = BasicAE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/BasicAE_UCSD.json")

elif model_used == "ProgAE":
    auto_encoder = ProgAE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/BasicAE_UCSD.json")

elif model_used == "AGE":
    auto_encoder = AGE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/AGE_UCSD.json")

elif model_used == "VAE":
    auto_encoder = VAE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/VAE_UCSD.json")

elif model_used == "GAN":
    auto_encoder = GAN(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/GAN_UCSD.json")

else:
    raise ValueError

ucsd_database = UCSDDatabase(database_path="datasets/ucsd/")
ucsd_database = ucsd_database.resized_to_scale(auto_encoder.input_shape)
ucsd_database.shuffle(seed=1)

max_scale = 4

auto_encoder.train(ucsd_database,
                   scale=max_scale,
                   epoch_length=250,
                   epochs=[100, 100, 100, 100, 300],
                   batch_size=[128, 128, 64, 32, 32],
                   pre_train=False)
