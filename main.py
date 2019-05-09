def main():
    # region Imports ...
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from tensorflow.python.keras import backend
    import tensorflow as tf
    import cv2
    from typing import Optional

    from models import BasicAE, VAE, GAN, VAEGAN, AGE
    from datasets import DatasetLoader, DatasetConfig
    from modalities import ModalityShape, RawVideo  # , OpticalFlow, DoG
    from utils.numpy_utils import NumpySeedContext
    # endregion

    # region Models/Datasets dictionaries
    models_dict = {"BasicAE": BasicAE,
                   "VAE": VAE,
                   "GAN": GAN,
                   "VAEGAN": VAEGAN,
                   "AGE": AGE,
                   }

    datasets_dict = {"UCSD_Ped2": ("ucsd/ped2", "UCSD_Ped"),
                     "UCSD_Ped1": ("ucsd/ped1", "UCSD_Ped"),
                     "Subway_Exit": ("subway/exit", "Subway"),
                     "Subway_Entrance": ("subway/entrance", "Subway"),
                     "Emoly": ("emoly", "Emoly")
                     }
    # endregion

    model_used = "BasicAE"
    dataset_used = "Emoly"
    alt_config_suffix_used = None
    use_flow = False
    use_patches = False
    previous_weights_to_load: Optional[str] = None
    previous_weights_to_load = "../logs/AutoEncoding-Anomalies/emoly/BasicAE/log_1557395333"

    # region Config/Dataset selection
    dataset_name, dataset_config_alias = datasets_dict[dataset_used]
    dataset_path = os.path.join("C:/datasets", dataset_name)
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

    # TODO : Load config for DatasetLoader
    # TODO : In config => Set ModalityShapes
    # TODO : In config => Set Normalization
    config = DatasetConfig(tfrecords_config_folder=dataset_path,
                           modalities_io_shapes=
                           {
                               RawVideo: ModalityShape(input_shape=(16, 128, 128, 1),
                                                       output_shape=(32, 128, 128, 1)),
                               # OpticalFlow: ModalityShape(input_shape=(32, 128, 128, 2),
                               #                            output_shape=(32, 128, 128, 2)),
                               # DoG: video_io_shape
                           },
                           output_range=auto_encoder.output_range)
    dataset = DatasetLoader(config=config)

    # endregion

    # region Test subset preview
    if preview_test_subset:
        _, videos = dataset.test_subset.get_batch(8, output_labels=False)
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
    session = tf.Session(config=config)
    backend.set_session(session)

    if previous_weights_to_load is not None:
        previous_weights_to_load: str = previous_weights_to_load
        print("=> Loading weights from :", previous_weights_to_load)
        auto_encoder.load_weights(previous_weights_to_load, epoch=26)
    # endregion

    with NumpySeedContext(seed=auto_encoder.seed):
        if profile:
            import cProfile

            print("===== Profiling activated ... =====")
            cProfile.run("auto_encoder.train(dataset, epoch_length=500, epochs=10, batch_size=6)", sort="cumulative")
        else:
            auto_encoder.train(dataset, dataset_name=dataset_name,
                               epoch_length=500, epochs=500, batch_size=16)


if __name__ == "__main__":
    # TODO : Use command line args
    # tf.enable_eager_execution()
    main()

# TODO : Type hints EVERYWHERE !

# TODO : Residual Scaling

# TODO : config - for callbacks (batch_size, samples count, ...)

# TODO : Make patches from images
