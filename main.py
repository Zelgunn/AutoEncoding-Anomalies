# region Select GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# endregion

# region Imports
import keras.backend as K
import tensorflow as tf
import cv2

from models import BasicAE, ProgAE, AGE, VAE, GAN, VAEGAN
from datasets import UCSDDatabase
from data_preprocessors import DropoutNoiser

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

model_used = "VAEGAN"
dataset_used = "UCSD_Ped2"
config_used = "configs/{model}_{dataset}.json".format(model=model_used, dataset=dataset_used)
preview_tensorboard_test_images = False
allow_gpu_growth = False

# region Model/Dataset initialization
# region Model
auto_encoder_class = models_dict[model_used]
auto_encoder = auto_encoder_class()
auto_encoder.build_model(config_used)
# endregion

# region Preprocessors
train_dropout_noise_rate = auto_encoder.config["data_generators"]["train"]["dropout_rate"]
train_preprocessors = [DropoutNoiser(inputs_dropout_rate=train_dropout_noise_rate)]

test_preprocessors = []
# endregion

# region Datasets
print("===== Loading data =====")
database_class, database_path = datasets_dict[dataset_used]
database = database_class(database_path=database_path,
                          train_preprocessors=train_preprocessors,
                          test_preprocessors=test_preprocessors)

print("===== Resizing data to input_shape =====")
database = database.resized_to_scale(auto_encoder.input_shape)

print("===== Normalizing data between {0} and {1} for activation \"{2}\"  =====".format(
    *auto_encoder.output_range, auto_encoder.output_activation))
database.normalize(auto_encoder.output_range[0], auto_encoder.output_range[1])

print("===== Shuffling data =====")
seed = 8
database.shuffle(seed=seed)
database.test_dataset.epoch_length = 2
# endregion
# endregion

# region Test dataset preview
if preview_tensorboard_test_images:
    samples_count = database.test_dataset.samples_count
    preview_count = auto_encoder.image_summaries_max_outputs
    previewed_image, _ = database.test_dataset.sample(preview_count, apply_preprocess_step=False, seed=0)
    previewed_image = (previewed_image - previewed_image.min()) / (previewed_image.max() - previewed_image.min())
    for i in range(preview_count):
        cv2.imshow("preview_{0} (seed={1})".format(i, seed), previewed_image[i])
    cv2.waitKey(120000)
    cv2.destroyAllWindows()
    exit()

# endregion

# region Session initialization
if allow_gpu_growth:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
# endregion

auto_encoder.train(database,
                   min_scale=4,
                   max_scale=4,
                   epoch_length=5,
                   epochs=[20, 20, 50, 50, 2000],
                   batch_size=[128, 128, 64, 32, 32],
                   pre_train=False)
