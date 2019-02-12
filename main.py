# region Select GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# endregion

# region Imports
import keras.backend as K
import tensorflow as tf
import cv2

from models import BasicAE, VAE, GAN, VAEGAN, AGE
from datasets import UCSDDatabase, SubwayDatabase
from data_preprocessors import DropoutNoiser

# endregion

# region Models/Datasets dictionaries
models_dict = {"BasicAE": BasicAE,
               "VAE": VAE,
               "GAN": GAN,
               "VAEGAN": VAEGAN,
               "AGE": AGE,
               }

datasets_dict = {"UCSD_Ped2": [UCSDDatabase, "../datasets/ucsd/ped2", "UCSD_Ped"],
                 "UCSD_Ped1": [UCSDDatabase, "../datasets/ucsd/ped1", "UCSD_Ped"],
                 "Subway_Exit": [SubwayDatabase, "../datasets/subway/exit", "Subway"],
                 "Subway_Entrance": [SubwayDatabase, "../datasets/subway/entrance", "Subway"]
                 }
# endregion

model_used = "VAEGAN"
dataset_used = "Subway_Exit"
database_class, database_path, database_config_alias = datasets_dict[dataset_used]
config_used = "configs/{dataset}/{model}_{dataset}.json".format(model=model_used, dataset=database_config_alias)
preview_tensorboard_test_images = False
allow_gpu_growth = False
profile = False

# region Model/Dataset initialization
# region Model
auto_encoder_class = models_dict[model_used]
auto_encoder = auto_encoder_class()
auto_encoder.build(config_used)
# endregion

# region Preprocessors
train_dropout_noise_rate = auto_encoder.config["data_generators"]["train"]["dropout_rate"]
train_preprocessors = [DropoutNoiser(inputs_dropout_rate=train_dropout_noise_rate)]

test_preprocessors = []
# endregion

# region Datasets
print("===== Loading data =====")
database = database_class(database_path=database_path,
                          train_preprocessors=train_preprocessors,
                          test_preprocessors=test_preprocessors)
database.load()

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
    previewed_image = database.test_dataset.sample_unprocessed_images(preview_count, seed=0,
                                                                      max_shard_count=preview_count)
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

if profile:
    import cProfile

    print("===== Profiling activated ... =====")
    cProfile.run("auto_encoder.train(database,\
                   min_scale=2,\
                   max_scale=3,\
                   epoch_length=2,\
                   epochs=[2, 2, 2, 2, 2],\
                   batch_size=[32, 32, 32, 32, 32],\
                   pre_train=True)", sort="cumulative")
else:
    # auto_encoder.train(database,
    #                    min_scale=2,
    #                    max_scale=3,
    #                    epoch_length=2,
    #                    epochs=[2, 2, 2, 2, 2],
    #                    batch_size=[32, 32, 32, 32, 32],
    #                    pre_train=True)
    auto_encoder.train(database,
                       min_scale=4,
                       max_scale=4,
                       epoch_length=200,
                       epochs=[20, 20, 50, 50, 2000],
                       batch_size=[32, 32, 32, 32, 32],
                       pre_train=False)

# TODO : Make patches from images
# TODO : Flow version
# TODO : Video version (Normal + Flow)
# TODO : Save complete model
