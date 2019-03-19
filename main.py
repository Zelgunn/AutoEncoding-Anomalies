# region Select GPU
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# endregion

# region Imports
from tensorflow.python.keras import backend
import tensorflow as tf
import cv2

from models import BasicAE, VAE, GAN, VAEGAN, AGE
from datasets import UCSDDataset, SubwayDataset

# endregion

# region Models/Datasets dictionaries
models_dict = {"BasicAE": BasicAE,
               "VAE": VAE,
               "GAN": GAN,
               "VAEGAN": VAEGAN,
               "AGE": AGE,
               }

datasets_dict = {"UCSD_Ped2": [UCSDDataset, "../datasets/ucsd/ped2", "UCSD_Ped"],
                 "UCSD_Ped1": [UCSDDataset, "../datasets/ucsd/ped1", "UCSD_Ped"],
                 "Subway_Exit": [SubwayDataset, "../datasets/subway/exit", "Subway"],
                 "Subway_Entrance": [SubwayDataset, "../datasets/subway/entrance", "Subway"]
                 }

# endregion

model_used = "BasicAE"
dataset_used = "UCSD_Ped2"
alt_config_suffix_used = None
use_flow = False
use_patches = False
previous_weights_to_load = None
# previous_weights_to_load = "../logs/AutoEncoding-Anomalies/UCSDDataset/BasicAE/"
# previous_weights_to_load = "../logs/AutoEncoding-Anomalies/UCSDDataset/VAE/"

# region Config/Dataset selection
dataset_class, dataset_path, dataset_config_alias = datasets_dict[dataset_used]
config_used = "configs/{dataset}/{model}_{dataset}.json".format(model=model_used, dataset=dataset_config_alias)
if alt_config_suffix_used is None:
    alt_config_used = None
else:
    alt_config_used = "configs/alt/{dataset}/{model}_{dataset}_{suffix}.json".format(
        model=model_used, dataset=dataset_config_alias, suffix=alt_config_suffix_used)
# endregion

preview_test_subset = False
allow_gpu_growth = False
profile = False

# region Model/Dataset initialization
# region Model
auto_encoder_class = models_dict[model_used]
auto_encoder = auto_encoder_class()
auto_encoder.image_summaries_max_outputs = 4
auto_encoder.load_config(config_used, alt_config_used)
auto_encoder.build_layers()
auto_encoder.compile()
# endregion

# region Print parameters
print("===============================================")
print("===== Running with following parameters : =====")
print("Model used \t\t\t\t\t:", model_used)
print("Dataset used \t\t\t\t\t:", dataset_used)
print("Use flow \t\t\t\t\t:", use_flow)
print("Use patches \t\t\t\t\t:", use_patches)
print("Dataset class \t\t\t\t\t:", str(dataset_class))
print("Dataset path \t\t\t\t\t:", dataset_path)
print("Dataset config alias \t\t\t\t:", dataset_config_alias)
print("Config used \t\t\t\t\t:", config_used)
print("Blocks used \t\t\t\t\t:", auto_encoder.block_type_name)
print("Total depth \t\t\t\t\t:", auto_encoder.compute_conv_depth())
print("Preview tensorboard test images \t\t:", preview_test_subset)
print("Allow GPU growth \t\t\t\t:", allow_gpu_growth)
print("Run cProfile \t\t\t\t\t:", profile)
print("===============================================")
# endregion

# region Datasets
print("===== Loading data =====")
dataset = dataset_class(dataset_path=dataset_path,
                        input_sequence_length=auto_encoder.input_sequence_length,
                        output_sequence_length=auto_encoder.output_sequence_length,
                        train_preprocessors=auto_encoder.train_data_preprocessors,
                        test_preprocessors=auto_encoder.test_data_preprocessors)
dataset.load()

print("===== Resizing data to input_shape =====")
dataset = auto_encoder.resized_dataset(dataset)

print("===== Normalizing data between {0} and {1} for activation \"{2}\"  =====".format(
    *auto_encoder.output_range, auto_encoder.output_activation["name"]))
dataset.normalize(*auto_encoder.output_range)
# endregion
# endregion

# region Test subset preview
if preview_test_subset:
    _, videos = dataset.test_subset.get_batch(8, seed=1, apply_preprocess_step=False, max_shard_count=8)
    for video in videos:
        for frame in video:
            cv2.imshow("frame", frame)
            cv2.waitKey(30)
    cv2.destroyAllWindows()
    exit()
# endregion

# region Session initialization
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = allow_gpu_growth
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)
backend.set_session(session)

if previous_weights_to_load is not None:
    previous_weights_to_load: str = previous_weights_to_load
    print("=> Loading weights from :", previous_weights_to_load)
    auto_encoder.load_weights(previous_weights_to_load, epoch=40)
# endregion

if profile:
    import cProfile

    print("===== Profiling activated ... =====")
    cProfile.run("auto_encoder.train(dataset, epoch_length=500, epochs=10, batch_size=6)", sort="cumulative")
else:
    auto_encoder.train(dataset, epoch_length=1000, epochs=500, batch_size=6)

# TODO : Residual Scaling

# TODO : config - for callbacks (batch_size, samples count, ...)

# TODO : Make patches from images
# TODO : Flow version
