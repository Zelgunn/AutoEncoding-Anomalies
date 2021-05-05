import tensorflow as tf
import numpy as np
import os
import json

from kitsune_main import KitsuneTest


def main():
    load_folder = "../logs/AEA/packet/Active Wiretap/train/viae/1616630794"
    config_path = os.path.join(load_folder, "main_config.json")
    weights_path = os.path.join(load_folder, "weights_032")

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    kitsune_test = KitsuneTest(dataset_name="Fuzzing", log_dir="../logs/AEA",
                               initial_epoch=32, config=config, load_weights=False)

    checkpoint = tf.train.Checkpoint(kitsune_test.model)
    checkpoint.restore(weights_path).expect_partial()

    packets = tf.convert_to_tensor(kitsune_test.get_packets()[:kitsune_test.train_samples_count], dtype=tf.float32)
    anomaly_scores = kitsune_test.compute_anomaly_scores(packets)

    anomaly_scores = anomaly_scores.numpy()
    np.save("../../tmp/anomaly_scores.npy", anomaly_scores)


if __name__ == "__main__":
    main()
