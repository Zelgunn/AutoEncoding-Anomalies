import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
import tensorflow as tf

from models import BasicAE, ProgAE, AGE, VAE, GAN, VAEGAN
from datasets import UCSDDatabase

models_dict = {"BasicAE": BasicAE,
               "ProgAE": ProgAE,
               "AGE": AGE,
               "VAE": VAE,
               "GAN": GAN,
               "VAEGAN": VAEGAN,
               }

datasets_dict = {"UCSD_Ped2": [UCSDDatabase, "datasets/ucsd/ped2"],
                 "UCSD_Ped1": [UCSDDatabase, "datasets/ucsd/ped1"]
                 }

model_used = "VAEGAN"
dataset_used = "UCSD_Ped2"
config_used = "configs/{model}_{dataset}.json".format(model=model_used, dataset=dataset_used)

auto_encoder_class = models_dict[model_used]
auto_encoder = auto_encoder_class()
auto_encoder.build_model(config_used)

database_class, database_path = datasets_dict[dataset_used]
database = database_class(database_path=database_path)
database = database.resized_to_scale(auto_encoder.input_shape)
database.shuffle(seed=1)

max_scale = 4

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.log_device_placement = False
# session = tf.Session(config=config)
# K.set_session(session)

auto_encoder.train(database,
                   scale=max_scale,
                   epoch_length=50,
                   epochs=[50, 50, 100, 200, 2000],
                   batch_size=[128, 128, 64, 32, 32],
                   pre_train=True)
