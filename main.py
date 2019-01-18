# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from models import BasicAE, ProgAE, AGE, VAE
from datasets import UCSDDatabase

model_used = "VAE"

if model_used == "BasicAE":
    auto_encoder = BasicAE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/BasicAE_UCSD.json")

elif model_used == "ProgAE":
    auto_encoder = BasicAE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/BasicAE_UCSD.json")

elif model_used == "AGE":
    auto_encoder = AGE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/AGE_UCSD.json")

elif model_used == "VAE":
    auto_encoder = VAE(image_summaries_max_outputs=3)
    auto_encoder.build_model("configs/VAE_UCSD.json")

else:
    raise ValueError

ucsd_database = UCSDDatabase(database_path="datasets/ucsd/")
ucsd_database = ucsd_database.resized_to_scale(auto_encoder.input_shape)
ucsd_database.shuffle(seed=0)

max_scale = 3

auto_encoder.train(ucsd_database,
                   scale=max_scale,
                   epoch_length=250,
                   epochs=[90, 100, 200, 1000],
                   batch_size=[128, 128, 64, 32])
