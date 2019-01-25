# region Select GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# endregion

# region Imports
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2

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
preview_tensorboard_test_images = False
allow_gpu_growth = False

# region Model/Dataset initialization
auto_encoder_class = models_dict[model_used]
auto_encoder = auto_encoder_class()
auto_encoder.build_model(config_used)

print("===== Loading data =====")
database_class, database_path = datasets_dict[dataset_used]
database = database_class(database_path=database_path)
print("===== Resizing data to input_shape =====")
database = database.resized_to_scale(auto_encoder.input_shape)
print("===== Normalizing data between {0} and {1} for activation \"{2}\"  =====".format(
    *auto_encoder.output_range, auto_encoder.output_activation))
database.normalize(auto_encoder.output_range[0], auto_encoder.output_range[1])
print("===== Shuffling data =====")
database.shuffle(seed=17)
# endregion

# region Test dataset preview
if preview_tensorboard_test_images:
    samples_count = database.test_dataset.samples_count
    preview_count = auto_encoder.image_summaries_max_outputs
    for i in range(preview_count):
        index = int(i * (samples_count - 1) / (preview_count - 1))
        previewed_image = database.test_dataset.images[index]
        previewed_image = (previewed_image - previewed_image.min()) / (previewed_image.max() - previewed_image.min())
        cv2.imshow("preview_{0}".format(index), previewed_image)
    cv2.waitKey(120000)
    cv2.destroyAllWindows()

# endregion

# region Session initialization
if allow_gpu_growth:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
# endregion

auto_encoder.train(database,
                   min_scale=3,
                   max_scale=4,
                   epoch_length=250,
                   epochs=[20, 20, 50, 50, 2000],
                   batch_size=[128, 128, 64, 32, 32],
                   pre_train=False)
