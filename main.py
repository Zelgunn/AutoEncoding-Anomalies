# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from models import BasicAE, ProgAE, AGE
from datasets import UCSDDatabase

auto_encoder = AGE(image_summaries_max_outputs=3)
auto_encoder.build_model("configs/AGE_UCSD.json")

ucsd_database = UCSDDatabase(database_path="datasets/ucsd/")
ucsd_database = ucsd_database.resized_to_scale(auto_encoder.input_shape)
ucsd_database.shuffle(seed=0)

max_scale = 3

auto_encoder.train(ucsd_database, scale=max_scale,
                   epoch_length=500, pre_train_epochs=100, epochs=250,
                   batch_size=64)
