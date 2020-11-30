# from tensorflow.keras.mixed_precision import experimental as mixed_precision
import argparse

from datasets.tfrecord_builders.SubwayTFRB import SubwayVideo
from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol, SubwayProtocol


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", default="ped2")
    arg_parser.add_argument("--mode", default="train")
    arg_parser.add_argument("--initial_epoch", default=None)
    arg_parser.add_argument("--log_dir", default="../logs/AEA")

    args = arg_parser.parse_args()
    dataset: str = args.dataset
    mode: str = args.mode
    initial_epoch = args.initial_epoch
    log_dir: str = args.log_dir

    best_weights = {
        "ped2": 16,
        "ped1": 50,
        "avenue": 22,
        "shanghaitech": 14,
        "exit": 11,
        "entrance": 97,
    }

    if initial_epoch is None:
        if mode == "train":
            initial_epoch = 0
        else:
            initial_epoch = best_weights[dataset]
    elif initial_epoch == "best":
        initial_epoch = best_weights[dataset]
    else:
        initial_epoch = int(initial_epoch)

    # policy = mixed_precision.Policy("mixed_float16")
    # mixed_precision.set_policy(policy)

    if dataset == "ped2":
        protocol = UCSDProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, dataset_version=2)
    elif dataset == "ped1":
        protocol = UCSDProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, dataset_version=1)
    elif dataset == "avenue":
        protocol = AvenueProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch)
    elif dataset == "shanghaitech":
        protocol = ShanghaiTechProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch)
    elif dataset == "exit":
        protocol = SubwayProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, video_id=SubwayVideo.EXIT)
    elif dataset == "entrance":
        protocol = SubwayProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, video_id=SubwayVideo.ENTRANCE)
    elif dataset == "mall1":
        protocol = SubwayProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, video_id=SubwayVideo.MALL1)
    elif dataset == "mall2":
        protocol = SubwayProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, video_id=SubwayVideo.MALL2)
    elif dataset == "mall3":
        protocol = SubwayProtocol(base_log_dir=log_dir, initial_epoch=initial_epoch, video_id=SubwayVideo.MALL3)
    else:
        raise ValueError(dataset)

    # root = r"..\datasets\ucsd\ped2"
    #
    # input_video = root + r"\Test\Test004"
    # output_video = root + r"\output.avi"
    #
    # protocol.autoencode_video(video_source=input_video,
    #                           target_path=output_video,
    #                           load_epoch=initial_epoch,
    #                           fps=25.0,
    #                           output_size=(128, 128))
    #
    # exit()

    if mode == "train":
        protocol.train_model()
    elif mode == "test":
        protocol.test_model()
    else:
        protocol.log_model_latent_codes(config=protocol.get_test_config())


if __name__ == "__main__":
    main()
