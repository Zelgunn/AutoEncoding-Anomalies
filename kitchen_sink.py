import numpy as np
import matplotlib.pyplot as plt


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
    # root += r"\ped1"
    # root += r"\ped2"
    # root += r"\avenue"
    root += r"\shanghaitech"
    # root += r"\subway_exit"
    # root += r"\subway_entrance"
    # root += r"\subway_entrance"
    # root += r"\subway_entrance"


if __name__ == "__main__":
    main()
