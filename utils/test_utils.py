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


def evaluate_model_anomaly_detection(model: Model,
                                     dataset: Dataset,
                                     epoch_length: int,
                                     batch_size: int,
                                     evaluate_on_whole_video: bool):
    dataset.epoch_length = epoch_length
    dataset.batch_size = batch_size

    input_layer = model.input
    pred_output = model.output
    true_output = tf.placeholder(dtype=tf.float32, shape=model.output_shape)
    error_op = tf.square(pred_output - true_output)

    if evaluate_on_whole_video:
        error_op = tf.reduce_sum(error_op, axis=[1, 2, 3, 4])
        predictions_shape = [epoch_length * batch_size]
    else:
        error_op = tf.reduce_sum(error_op, axis=[2, 3, 4])
        predictions_shape = [epoch_length * batch_size, dataset.output_sequence_length]

    errors = np.zeros(shape=predictions_shape)
    labels = np.zeros(shape=predictions_shape, dtype=np.bool)
    session = K.get_session()
    for i in tqdm(range(dataset.epoch_length), desc="Computing errors..."):
        images, step_labels, _ = dataset.sample(return_labels=True)
        x, y_true = dataset.divide_batch_io(images)
        x, y_true = dataset.apply_preprocess(x, y_true)

        step_error = session.run(error_op, feed_dict={input_layer: x, true_output: y_true})
        indices = np.arange(i * dataset.batch_size, (i + 1) * dataset.batch_size)

        errors[indices] = step_error
        if evaluate_on_whole_video:
            labels[indices] = np.any(step_labels, axis=1)
        else:
            labels[indices] = step_labels

    errors = (errors - errors.min()) / (errors.max() - errors.min())
    anomaly_percentage = np.mean(labels.astype(np.float32))

    labels = tf.constant(labels)
    errors = tf.constant(errors)
    roc_ops = tf.metrics.auc(labels, errors, summation_method="careful_interpolation")
    pr_ops = tf.metrics.auc(labels, errors, curve="PR", summation_method="careful_interpolation")
    auc_ops = roc_ops + pr_ops

    session.run(tf.local_variables_initializer())
    _, roc, _, pr = session.run(auc_ops)
    print("ROC : {} | PR : {} | Anomaly percentage in samples : {}".format(roc, pr, anomaly_percentage))
    return roc, pr
