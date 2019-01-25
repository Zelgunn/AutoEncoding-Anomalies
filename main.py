# region Select GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# endregion

# region Imports
import keras.backend as K
import tensorflow as tf

from models import BasicAE, ProgAE, AGE, VAE, GAN, VAEGAN
from datasets import UCSDDatabase

# endregion

# region Models/Datasets dictionaries
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
# endregion

model_used = "GAN"
dataset_used = "UCSD_Ped2"
config_used = "configs/{model}_{dataset}.json".format(model=model_used, dataset=dataset_used)

# region Model/Dataset initialization
auto_encoder_class = models_dict[model_used]
auto_encoder = auto_encoder_class()
auto_encoder.build_model(config_used)

database_class, database_path = datasets_dict[dataset_used]
database = database_class(database_path=database_path)
database = database.resized_to_scale(auto_encoder.input_shape)
database.normalize(auto_encoder.output_range[0], auto_encoder.output_range[1])
database.shuffle(seed=1)
# endregion

# region Session initialization
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
session = tf.Session(config=config)
K.set_session(session)
# endregion

auto_encoder.train(database,
                   min_scale=3,
                   max_scale=4,
                   epoch_length=200,
                   epochs=[20, 20, 50, 50, 2000],
                   batch_size=[128, 128, 64, 32, 32],
                   pre_train=False)
