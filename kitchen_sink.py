import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_labels(labels: np.ndarray):
    start = -1

    for i in range(len(labels)):
        if start == -1:
            if labels[i]:
                start = i
        else:
            if not labels[i]:
                plt.gca().axvspan(start, i, alpha=0.25, color="red")
                start = -1

    if start != -1:
        plt.gca().axvspan(start, len(labels) - 1, alpha=0.25, color="red")


def process(pred, start, end):
    pred_window = pred[start:end]
    pred_window = 1.0 - pred_window
    pred_window = np.repeat(pred_window, 32)
    return pred_window


def window_normalization(inputs, window_size):
    if window_size < 0:
        return inputs
    outputs = np.zeros_like(inputs)
    for i in range(len(inputs)):
        start = max(i - window_size // 2, 0)
        end = i + window_size // 2
        window = inputs[start: end]
        outputs[i] = (inputs[i] - window.min()) / (window.max() - window.min())
    return outputs


def main():
    root = r"D:\Users\Degva\Documents\_PhD\Tensorflow\logs\AEA\protocols\video"
    # root += r"\ped1\anomaly_detection\1572958822_iae"
    # root += r"\ped2\anomaly_detection\1572953851_iae"
    # root += r"\avenue\anomaly_detection\1572961348_iae"
    # root += r"\shanghaitech\anomaly_detection\1572962968_iae"
    root += r"\subway_entrance\anomaly_detection\1573563762_iae_ours"
    # root += r"\subway_exit\anomaly_detection\1572956839_iae"

    pred_path = os.path.join(root, "predictions.npy")
    labels_path = os.path.join(root, "labels.npy")

    predictions = np.load(pred_path)
    labels = np.load(labels_path)[0]

    from anomaly_detection import AnomalyDetector

    anomaly_detector = AnomalyDetector(autoencoder=None,
                                       output_length=64,
                                       )
    anomaly_detector.anomaly_metrics_names = ["mse", "mae", "ssim", "psnr",
                                              "interpolation_mse", "interpolation_mae",
                                              "latent_code_surprisal"]

    results = anomaly_detector.evaluate_predictions(predictions=predictions, labels=labels)

    for i in range(anomaly_detector.metric_count):
        metric_results_string = "Anomaly_score ({}):".format(anomaly_detector.anomaly_metrics_names[i])
        for result_name, result_values in results.items():
            metric_results_string += " {} = {} |".format(result_name, result_values[i])
        print(metric_results_string)

    predictions = predictions[4:5]
    anomaly_detector.anomaly_metrics_names = ["interpolation_mse"]
    anomaly_detector.plot_sample_predictions(predictions, labels, root, "main")

    additional_config = {"epoch": 97,
                         "model_name": "iae",
                         "stride": 1,
                         "pre-normalize predictions": True,
                         "sensitivity": 1
                         }

    anomaly_detector.save_evaluation_results(log_dir=root,
                                             results=results,
                                             additional_config=additional_config)


if __name__ == "__main__":
    main()
