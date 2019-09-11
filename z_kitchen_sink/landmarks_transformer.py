import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import os
from time import time

from callbacks import AUCCallback, LandmarksVideoCallback
from modalities import ModalityLoadInfo, Landmarks, Pattern
from anomaly_detection import RawPredictionsModel, AnomalyDetector
from utils.train_utils import save_model_info
from transformers import Transformer
from transformers.core import PositionalEncodingMode
from z_kitchen_sink.utils import get_landmarks_datasets


def add_batch_noise(*args):
    inputs, outputs = args[0]
    noise = tf.random.uniform([], minval=-0.1, maxval=0.1)
    inputs = tf.clip_by_value(inputs + noise, clip_value_min=0.0, clip_value_max=1.0)
    outputs = tf.clip_by_value(outputs + noise, clip_value_min=0.0, clip_value_max=1.0)
    return (inputs, outputs),


def train_landmarks_transformer():
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/landmarks_transformer"
    batch_size = 16
    input_length = 32
    output_length = input_length + 16
    frame_per_prediction = 4

    landmarks_transformer = Transformer(max_input_length=input_length,
                                        max_output_length=output_length // frame_per_prediction,
                                        input_size=68 * 2,
                                        output_size=68 * 2 * frame_per_prediction,
                                        output_activation="linear",
                                        layers_intermediate_size=32,
                                        layers_count=4,
                                        attention_heads_count=4,
                                        attention_key_size=16,
                                        attention_values_size=16,
                                        dropout_rate=0.0,
                                        # decoder_pre_net=pre_net,
                                        decoder_pre_net=None,
                                        positional_encoding_mode=PositionalEncodingMode.ADD,
                                        positional_encoding_range=0.1,
                                        positional_encoding_size=32,
                                        name="Transformer")

    landmarks_transformer.add_transformer_loss()

    optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.98)
    landmarks_transformer.compile(optimizer)
    landmarks_transformer.summary()

    landmarks_transformer.load_weights(base_log_dir + "/weights_011.hdf5")

    evaluator = landmarks_transformer.make_evaluator(output_length)
    optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.98)
    evaluator.compile(optimizer, loss="mse")

    # region Datasets
    dataset_loader, train_subset, test_subset = get_landmarks_datasets()

    pattern = Pattern(
        ModalityLoadInfo(Landmarks, input_length, (input_length, 136)),
        ModalityLoadInfo(Landmarks, output_length, (output_length // frame_per_prediction,
                                                    136 * frame_per_prediction))
    )
    anomaly_pattern = Pattern(
        *pattern,
        "labels"
    )

    train_dataset = train_subset.make_tf_dataset(pattern.with_added_depth())
    train_dataset = train_dataset.map(add_batch_noise)
    test_dataset = test_subset.make_tf_dataset(pattern.with_added_depth())
    # endregion

    # tmp = Model(inputs=landmarks_transformer.inputs,
    #             outputs=landmarks_transformer.decoder_attention_weights)
    # self_pwet, encoder_pwet = tmp.predict(train_dataset.batch(1), steps=1)
    # self_pwet = np.squeeze(self_pwet)
    # encoder_pwet = np.squeeze(encoder_pwet)
    # layer_count, head_count, _, _ = self_pwet.shape
    # import matplotlib.pyplot as plt
    # for i in range(layer_count):
    #     for j in range(head_count):
    #         plt.pcolormesh(self_pwet[i, j], cmap="jet")
    #         plt.show()
    #         plt.pcolormesh(encoder_pwet[i, j], cmap="jet")
    #         plt.show()
    # exit()

    # region Callbacks
    log_dir = os.path.join(base_log_dir, "log_{}".format(int(time())))
    os.makedirs(log_dir)
    save_model_info(landmarks_transformer, log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, update_freq="epoch", profile_batch=0)

    # region Landmarks (video)
    # region Default
    lvc_train_dataset = train_dataset.map(lambda data: (data, data[1]))
    lvc_test_dataset = test_dataset.map(lambda data: (data, data[1]))

    train_landmarks_callback = LandmarksVideoCallback(lvc_train_dataset, landmarks_transformer, tensorboard,
                                                      is_train_callback=True, fps=25)
    test_landmarks_callback = LandmarksVideoCallback(lvc_test_dataset, landmarks_transformer, tensorboard,
                                                     is_train_callback=False)
    # endregion

    # region Autonomous
    lvc_train_dataset = train_dataset.map(lambda data: data)
    lvc_test_dataset = test_dataset.map(lambda data: data)
    autonomous_train_landmarks_callback = LandmarksVideoCallback(lvc_train_dataset, evaluator,
                                                                 tensorboard, is_train_callback=True,
                                                                 prefix="autonomous")
    autonomous_test_landmarks_callback = LandmarksVideoCallback(lvc_test_dataset, evaluator,
                                                                tensorboard, is_train_callback=False,
                                                                prefix="autonomous")
    # endregion
    # endregion

    # region AUC
    # region Default (32 frames)
    # raw_predictions = RawPredictionsLayer(output_length=input_length)([landmarks_transformer.output,
    #                                                                    landmarks_transformer.decoder_target_layer])
    # raw_predictions_model = Model(inputs=landmarks_transformer.inputs,
    #                               outputs=raw_predictions,
    #                               name="raw_predictions_model_{}".format(input_length))
    #
    # auc_callback_32 = make_auc_callback(test_subset=test_subset,
    #                                     pattern=anomaly_pattern,
    #                                     predictions_model=raw_predictions_model,
    #                                     tensorboard=tensorboard,
    #                                     samples_count=512,
    #                                     prefix=str(input_length))
    # endregion
    # region Default (128 frames)
    raw_predictions_model = RawPredictionsModel(landmarks_transformer, output_length)

    auc_callback_128 = AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                               test_subset=test_subset, pattern=anomaly_pattern, samples_count=512,
                                               prefix=str(output_length))
    # endregion
    # region Evaluator
    # evaluator_ground_truth = Input(batch_shape=evaluator.output.shape, name="EvaluatorGroundTruth")
    # evaluator_raw_predictions = RawPredictionsLayer()([evaluator.output, evaluator_ground_truth])
    evaluator_raw_predictions_model = RawPredictionsModel(landmarks_transformer, output_length)
    evaluator_auc_callback = AUCCallback.from_subset(predictions_model=evaluator_raw_predictions_model,
                                                     tensorboard=tensorboard, test_subset=test_subset,
                                                     pattern=anomaly_pattern, samples_count=512, prefix="Evaluator")
    # endregion
    # endregion

    model_checkpoint = ModelCheckpoint(os.path.join(base_log_dir, "weights.{epoch:03d}.hdf5", ))

    callbacks = [model_checkpoint,
                 tensorboard,
                 train_landmarks_callback,
                 test_landmarks_callback,
                 autonomous_train_landmarks_callback,
                 autonomous_test_landmarks_callback,
                 # auc_callback_32,
                 auc_callback_128,
                 evaluator_auc_callback
                 ]

    # endregion

    train_dataset = train_dataset.batch(batch_size).prefetch(-1)
    test_dataset = test_dataset.batch(batch_size)

    # evaluator_train_dataset = train_dataset.map(lambda x: x)
    # evaluator_test_dataset = test_dataset.map(lambda x: x)

    landmarks_transformer.fit(train_dataset, epochs=50, steps_per_epoch=25000,
                              validation_data=test_dataset, validation_steps=2000,
                              callbacks=callbacks)

    # evaluator.fit(evaluator_train_dataset, epochs=30, steps_per_epoch=2000,
    #               validation_data=evaluator_test_dataset, validation_steps=500,
    #               callbacks=callbacks)

    anomaly_detector = AnomalyDetector(autoencoder=landmarks_transformer,
                                       output_length=output_length)
    anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                          pattern=anomaly_pattern.with_added_depth(),
                                          log_dir=log_dir,
                                          stride=1,
                                          pre_normalize_predictions=True)
