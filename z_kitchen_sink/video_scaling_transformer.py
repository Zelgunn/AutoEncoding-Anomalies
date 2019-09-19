import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import cv2
from time import time
from typing import List

from callbacks import ImageCallback, AUCCallback
from datasets.loaders import DatasetConfig, DatasetLoader
from modalities import ModalityLoadInfo, RawVideo, Pattern
from anomaly_detection import RawPredictionsLayer, AnomalyDetector
from utils.train_utils import save_model_info
from transformers import VideoTransformer


def train_scaling_video_transformer():
    # tf.keras.mixed_precision.experimental.set_policy("infer_float32_vars")

    dataset_name = "emoly"
    batch_size = 2
    length = 16
    height = 64
    width = 64
    channel_count = 1 if dataset_name is "ucsd" else 3
    what_to_do = "train"
    initial_epoch = 0

    subscale_configs = {
        "temporal":
            {
                "subscale_stride": (16, 1, 1),
                "block_shapes": [
                    (1, height // 4, 8),
                    (1, height, 2),
                    (1, 2, width),
                    (1, 8, width // 4)
                ],
                "copy_regularization_factor": 0.0
            },
        "spatial":
            {
                "subscale_stride": (1, 4, 4),
                "block_shapes": [
                    (16, 4, width // 32),
                    (2, height // 4, 4),
                    (2, 4, width // 4),
                    (16, height // 32, 4)
                ],
                "copy_regularization_factor": 1.0
            },
        "spatio_temporal":
            {
                "subscale_stride": (4, 2, 2),
                "block_shapes": [
                    (4, 8, width // 16),
                    (1, height // 2, 4),
                    (1, 4, width // 2),
                    (4, height // 16, 8)
                ],
                "copy_regularization_factor": 1.0
            }
    }
    subscale_config_used = subscale_configs["spatial"]

    video_transformer = VideoTransformer(input_shape=(length, height, width, channel_count),
                                         subscale_stride=subscale_config_used["subscale_stride"],
                                         embedding_size=128,
                                         hidden_size=256,
                                         block_shapes=subscale_config_used["block_shapes"],
                                         attention_head_count=4,
                                         attention_head_size=128,
                                         copy_regularization_factor=subscale_config_used["copy_regularization_factor"],
                                         positional_encoding_range=0.1
                                         )

    # region Patterns
    # def reduce_channels(video: tf.Tensor):
    #     return tf.reduce_mean(video, axis=-1, keepdims=True)

    def preprocess_video(video: tf.Tensor, labels: tf.Tensor = None):
        # video = reduce_channels(video)

        if what_to_do == "train":
            crop_ratio = tf.random.uniform(shape=(), minval=0.75, maxval=1.0)
            original_shape = tf.cast(tf.shape(video), tf.float32)
            original_height, original_width = original_shape[1], original_shape[2]
            crop_size = [length, crop_ratio * original_height, crop_ratio * original_width, channel_count]
            video = tf.image.random_crop(video, crop_size)
            video = tf.image.resize(video, (height, width))
            video = tf.image.random_brightness(video, max_delta=0.2)
        else:
            video = tf.image.resize(video, (height, width))

        if labels is not None:
            return video, labels

        return video

    pattern = Pattern(
        ModalityLoadInfo(RawVideo, length),
        output_map=preprocess_video
    )
    anomaly_pattern = Pattern(*pattern, "labels", output_map=preprocess_video)
    # endregion

    # region Datasets
    dataset_path = "../datasets/ucsd/ped2" if dataset_name is "ucsd" else "../datasets/emoly"
    dataset_config = DatasetConfig(dataset_path, output_range=(0.0, 1.0))
    dataset_loader = DatasetLoader(config=dataset_config)

    dataset = dataset_loader.train_subset.make_tf_dataset(pattern)
    dataset = dataset.batch(batch_size)

    dataset_loader.test_subset.subset_folders = [folder for folder in dataset_loader.test_subset.subset_folders
                                                 if "acted" in folder]

    validation_dataset = dataset_loader.test_subset.make_tf_dataset(pattern)
    validation_dataset = validation_dataset.batch(batch_size)
    # endregion

    # region Log dir
    base_log_dir = "../logs/AutoEncoding-Anomalies/kitchen_sink/video_transformer/{}".format(dataset_name)
    base_log_dir = os.path.normpath(base_log_dir)
    weights_name = "weights_{epoch:03d}.hdf5"
    dir_name = "log_{}" if what_to_do is "train" else "test_log_{}"
    log_dir = os.path.join(base_log_dir, dir_name.format(int(time())))
    if what_to_do is not "show":
        os.makedirs(log_dir)
    save_model_info(video_transformer.trainer, log_dir)
    weights_path = os.path.join(log_dir, weights_name)
    # endregion

    if initial_epoch > 0:
        video_transformer.trainer.load_weights(os.path.join(base_log_dir, weights_name.format(epoch=initial_epoch)))

    if what_to_do == "train":
        # region Callbacks
        callbacks: List[tf.keras.callbacks.Callback] = []

        tensorboard = TensorBoard(log_dir=log_dir, profile_batch=0)
        callbacks.append(tensorboard)

        model_checkpoint = ModelCheckpoint(weights_path, verbose=1)
        callbacks.append(model_checkpoint)

        # region Image callbacks
        train_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=video_transformer,
                                                                    subset=dataset_loader.train_subset,
                                                                    pattern=pattern,
                                                                    name="train",
                                                                    is_train_callback=True,
                                                                    tensorboard=tensorboard,
                                                                    epoch_freq=1)
        test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=video_transformer,
                                                                   subset=dataset_loader.test_subset,
                                                                   pattern=pattern,
                                                                   name="test",
                                                                   is_train_callback=False,
                                                                   tensorboard=tensorboard,
                                                                   epoch_freq=1)
        alt_video_transformer = video_transformer.model_using_decoder_outputs
        alt_test_image_callbacks = ImageCallback.from_model_and_subset(autoencoder=alt_video_transformer,
                                                                       subset=dataset_loader.test_subset,
                                                                       pattern=pattern,
                                                                       name="test_udo",
                                                                       is_train_callback=False,
                                                                       tensorboard=tensorboard,
                                                                       epoch_freq=1)

        image_callbacks = train_image_callbacks + test_image_callbacks + alt_test_image_callbacks
        callbacks += image_callbacks

        # endregion
        # region AUC
        def make_transformer_auc_callback(auc_mode: str):
            if auc_mode == "udo":
                output_used = video_transformer.model_using_decoder_outputs.output
            else:
                output_used = video_transformer.output

            raw_predictions = RawPredictionsLayer(output_length=length)([output_used, video_transformer.input])
            raw_predictions_model = Model(inputs=video_transformer.inputs,
                                          outputs=raw_predictions,
                                          name="{}_raw_predictions_model".format(auc_mode))

            return AUCCallback.from_subset(predictions_model=raw_predictions_model, tensorboard=tensorboard,
                                           test_subset=dataset_loader.test_subset, pattern=anomaly_pattern,
                                           samples_count=128, epoch_freq=1, batch_size=4, prefix=auc_mode)

        ugt_auc_callback = make_transformer_auc_callback("ugt")
        callbacks.append(ugt_auc_callback)
        udo_auc_callback = make_transformer_auc_callback("udo")
        callbacks.append(udo_auc_callback)
        # endregion
        # endregion

        video_transformer.fit(dataset,
                              batch_size=batch_size,
                              epochs=100,
                              steps_per_epoch=200,
                              initial_epoch=initial_epoch,
                              validation_data=validation_dataset,
                              validation_steps=20,
                              callbacks=callbacks)
    elif what_to_do == "show":
        video_transformer.trainer.summary()

        for i, batch in zip(range(10), validation_dataset):
            predicted_easy = video_transformer(batch[:1], use_decoder_outputs=False)
            predicted_hard = video_transformer(batch[:1], use_decoder_outputs=True)
            for k in range(16):
                predicted_frame_easy = predicted_easy.numpy()[0, k]
                predicted_frame_hard = predicted_hard.numpy()[0, k]
                predicted_frame_easy = cv2.resize(predicted_frame_easy, (256, 256), interpolation=cv2.INTER_NEAREST)
                predicted_frame_hard = cv2.resize(predicted_frame_hard, (256, 256), interpolation=cv2.INTER_NEAREST)
                true_frame = cv2.resize(batch.numpy()[0, k], (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("predicted_easy", predicted_frame_easy)
                cv2.imshow("predicted_hard", predicted_frame_hard)
                cv2.imshow("frame", true_frame)
                cv2.waitKey(0)
    elif what_to_do == "anomaly_detection":
        detector_metrics = ["mse", "ssim", "mae"]
        detector_stride = 4
        pre_normalize_predictions = True
        use_decoder_outputs = True
        model_used = video_transformer.model_using_decoder_outputs if use_decoder_outputs else video_transformer

        anomaly_detector = AnomalyDetector(autoencoder=model_used,
                                           output_length=length,
                                           metrics=detector_metrics)

        anomaly_detector.predict_and_evaluate(dataset=dataset_loader,
                                              pattern=anomaly_pattern.with_added_depth().with_added_depth(),
                                              log_dir=log_dir,
                                              stride=detector_stride,
                                              pre_normalize_predictions=pre_normalize_predictions,
                                              additional_config={
                                                  "initial_epoch": initial_epoch,
                                                  "use_decoder_outputs": use_decoder_outputs,
                                                  "subscale_config_used": str(subscale_config_used)
                                              }
                                              )
    elif what_to_do == "generate":
        pass
