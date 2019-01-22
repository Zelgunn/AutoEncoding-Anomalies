import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
import tensorflow as tf

from models import BasicAE, ProgAE, AGE, VAE, GAN, VAEGAN
from datasets import UCSDDatabase

model_used = "VAEGAN"

if model_used == "BasicAE":
    auto_encoder = BasicAE()
    auto_encoder.build_model("configs/BasicAE_UCSD.json")

elif model_used == "ProgAE":
    auto_encoder = ProgAE()
    auto_encoder.build_model("configs/BasicAE_UCSD.json")

elif model_used == "AGE":
    auto_encoder = AGE()
    auto_encoder.build_model("configs/AGE_UCSD.json")

elif model_used == "VAE":
    auto_encoder = VAE()
    auto_encoder.build_model("configs/VAE_UCSD.json")

elif model_used == "GAN":
    auto_encoder = GAN()
    auto_encoder.build_model("configs/GAN_UCSD.json")

elif model_used == "VAEGAN":
    auto_encoder = VAEGAN()
    auto_encoder.build_model("configs/VAEGAN_UCSD.json")

else:
    raise ValueError

ucsd_database = UCSDDatabase(database_path="datasets/ucsd/")
ucsd_database = ucsd_database.resized_to_scale(auto_encoder.input_shape)
ucsd_database.shuffle(seed=1)

max_scale = 4

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.log_device_placement = False
session = tf.Session(config=config)
K.set_session(session)

auto_encoder.train(ucsd_database,
                   scale=max_scale,
                   epoch_length=250,
                   epochs=[50, 50, 250, 250, 2000],
                   batch_size=[128, 128, 64, 32, 32],
                   pre_train=True)
