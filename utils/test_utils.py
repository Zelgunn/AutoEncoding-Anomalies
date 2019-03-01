import tensorflow as tf
from keras.models import Model
import keras.backend as K
import cv2
import numpy as np
from tqdm import tqdm

from datasets import Dataset


def visualize_model_errors(model: Model, dataset: Dataset, images_size=(512, 512), fps=25):
    key = 13
    escape_key = 27
    seed = 0
    i = 0
    fps = int(1000 / fps)
    y_pred = y_true = max_error = None

    while key != escape_key:
        if key != -1:
            seed += 1
            i = 0
            x, y_true = dataset.get_batch(seed=seed)
            y_pred = model.predict(x)
            max_error = np.abs(y_pred - y_true).max()

        y_pred_resized = cv2.resize(y_pred[0][i], dsize=images_size)
        y_true_resized = cv2.resize(y_true[0][i], dsize=images_size)
        error = np.abs(y_pred_resized - y_true_resized) / max_error

        composite = np.zeros(shape=images_size, dtype=np.float32)
        composite = np.repeat(composite[:, :, np.newaxis], 3, axis=2)

        composite[..., 0] = (1.0 - error) * 90
        composite[..., 1] = error
        composite[..., 2] = y_true_resized
        composite = cv2.cvtColor(composite, cv2.COLOR_HSV2BGR)

        cv2.imshow("y_true", y_true_resized)
        cv2.imshow("y_pred", y_pred_resized)
        cv2.imshow("error", error)
        cv2.imshow("composite", composite)
        key = cv2.waitKey(fps)

        i = (i + 1) % dataset.output_sequence_length
    cv2.destroyAllWindows()


def evaluate_model_anomaly_detection_on_dataset(model: Model,
                                                dataset: Dataset,
                                                stride=None,
                                                variational_resampling=0):
    roc_values, pr_values, anomaly_percentages = [], [], []
    for i in range(dataset.videos_count):
        roc, pr, anomaly_percentage = evaluate_model_anomaly_detection_one_video(model, dataset, i, stride,
                                                                                 variational_resampling)
        roc_values.append(roc)
        pr_values.append(pr)
        anomaly_percentages.append(anomaly_percentage)

        print("{} => ROC : {} | PR : {} | Anomaly percentage in samples : {}".format(i, roc, pr, anomaly_percentage))

    roc = np.mean(roc_values)
    pr = np.mean(pr_values)
    anomaly_percentage = np.mean(anomaly_percentages)
    print("Global => ROC : {} | PR : {} | Anomaly percentage in samples : {}".format(roc, pr, anomaly_percentage))


def evaluate_model_anomaly_detection_one_video(model: Model,
                                               dataset,
                                               video_index: int,
                                               stride=None,
                                               variational_resampling=0):
    input_sequence_length = model.input_shape[1]
    output_sequence_length = model.output_shape[1]
    if stride is None:
        stride = input_sequence_length

    # region Prediction graph
    input_layer = model.input
    pred_output = model.output
    true_output = tf.placeholder(dtype=tf.float32, shape=model.output_shape)
    predictions_op = tf.square(pred_output - true_output)
    predictions_op = tf.reduce_sum(predictions_op, axis=[2, 3, 4])
    # endregion

    steps_count = (dataset.get_video_length(video_index) - output_sequence_length) // stride
    predictions_shape = [steps_count * output_sequence_length, dataset.output_sequence_length]

    predictions = np.zeros(shape=predictions_shape)
    labels = np.zeros(shape=predictions_shape, dtype=np.bool)
    session = K.get_session()
    for i in tqdm(range(steps_count), desc="Computing errors..."):
        start = i * stride
        end = start + output_sequence_length
        videos = dataset.get_video_frames(video_index, start, end)
        videos = np.expand_dims(videos, axis=0)
        step_labels = dataset.get_video_frame_labels(video_index, start, end)

        x, y_true = dataset.divide_batch_io(videos)

        feed_dict = {input_layer: x, true_output: y_true}
        step_predictions = session.run(predictions_op, feed_dict)
        if variational_resampling > 0:
            for _ in range(variational_resampling):
                step_predictions += session.run(predictions_op, feed_dict)
            step_predictions /= (variational_resampling + 1)

        indices = np.arange(i * output_sequence_length, (i + 1) * output_sequence_length)
        predictions[indices] = step_predictions
        labels[indices] = step_labels

    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())
    anomaly_percentage = np.mean(labels.astype(np.float32))

    # region AUC (ROC & PR) graph
    labels = tf.constant(labels)
    predictions = tf.constant(predictions)
    roc_ops = tf.metrics.auc(labels, predictions, summation_method="careful_interpolation")
    pr_ops = tf.metrics.auc(labels, predictions, curve="PR", summation_method="careful_interpolation")
    auc_ops = roc_ops + pr_ops
    # endregion

    session.run(tf.local_variables_initializer())
    _, roc, _, pr = session.run(auc_ops)
    return roc, pr, anomaly_percentage
