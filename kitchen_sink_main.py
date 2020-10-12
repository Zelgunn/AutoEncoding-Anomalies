import numpy as np
import matplotlib.pyplot as plt
import os

from anomaly_detection import AnomalyDetector


def load_predictions(log_id):
    log_id = str(log_id)
    root = r"C:\Users\Degva\Documents\_PhD\Tensorflow\logs\AEA\video\ped2\anomaly_detection\led"

    mae = np.load(os.path.join(root, log_id, "predictions_1.npy"))
    led = np.load(os.path.join(root, log_id, "predictions_4.npy"))
    labels = np.load(os.path.join(root, log_id, "labels.npy"), allow_pickle=True)

    mae = mae.mean(axis=- 1)
    mae = (mae - mae.min()) / (mae.max() - mae.min())

    labels = np.concatenate(labels, axis=0)

    return mae, led, labels


def combine(mae, led, w_mae, w_led):
    combined = mae * w_mae + led * w_led
    combined = combined / combined.max()
    return combined


def plot_results(mae, led, combined, labels):
    plt.plot(mae)
    plt.plot(led)
    plt.plot(combined)

    start = -1

    for i in range(len(labels)):
        if start == -1:
            if labels[i]:
                start = i
        else:
            if not labels[i]:
                plt.gca().axvspan(start, i, alpha=0.25, color="red", linewidth=0)
                start = -1

    if start != -1:
        plt.gca().axvspan(start, len(labels) - 1, alpha=0.25, color="red", linewidth=0)

    plt.legend(["mae", "led", "combined"])

    plt.show()


def main():
    mae, led, labels = load_predictions(1601803868)
    # for i in range(20):
    #     w_mae = min((i + 1) / 10.0, 1.0)
    #     w_led = min((20 - i) / 10.0, 1.0)
    #     combined = combine(mae, led, w_mae, w_led)
    #     results = AnomalyDetector.evaluate_metric_predictions(combined, labels)
    #     print("=" * 20, i, w_mae, w_led, "=" * 20)
    #     for result_name, result_value in results.items():
    #         print(result_name, " : ", float(result_value))
    combined = combine(mae, led, w_mae=1.0, w_led=1.0)
    results = AnomalyDetector.evaluate_metric_predictions(combined, labels)
    for result_name, result_value in results.items():
        print(result_name, " : ", float(result_value))
    plot_results(mae, led, combined, labels)


if __name__ == "__main__":
    main()
