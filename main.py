# from tensorflow.keras.mixed_precision import experimental as mixed_precision
import argparse

from datasets.tfrecord_builders.SubwayTFRB import SubwayVideo
from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol, SubwayProtocol
from protocols.packet_protocols import KitsuneProtocol
from protocols.audio_video_protocols import EmolyProtocol


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", default="ped2")
    arg_parser.add_argument("--mode", default="train")
    arg_parser.add_argument("--epoch", default=None)
    arg_parser.add_argument("--log_dir", default="../logs/AEA")

    args = arg_parser.parse_args()
    run_protocol(dataset=args.dataset, mode=args.mode, epoch=args.epoch, log_dir=args.log_dir)


def run_protocol(dataset: str, mode: str, epoch: int, log_dir: str):
    if epoch is None:
        if mode == "train":
            epoch = 0
        else:
            raise ValueError("Epoch must be known for testing, got None.")
    else:
        epoch = int(epoch)

    # policy = mixed_precision.Policy("mixed_float16")
    # mixed_precision.set_policy(policy)

    if dataset == "ped2":
        protocol = UCSDProtocol(base_log_dir=log_dir, epoch=epoch, dataset_version=2)
    elif dataset == "ped1":
        protocol = UCSDProtocol(base_log_dir=log_dir, epoch=epoch, dataset_version=1)
    elif dataset == "avenue":
        protocol = AvenueProtocol(base_log_dir=log_dir, epoch=epoch)
    elif dataset == "shanghaitech":
        protocol = ShanghaiTechProtocol(base_log_dir=log_dir, epoch=epoch)
    elif dataset == "exit":
        protocol = SubwayProtocol(base_log_dir=log_dir, epoch=epoch, video_id=SubwayVideo.EXIT)
    elif dataset == "entrance":
        protocol = SubwayProtocol(base_log_dir=log_dir, epoch=epoch, video_id=SubwayVideo.ENTRANCE)
    elif dataset == "mall1":
        protocol = SubwayProtocol(base_log_dir=log_dir, epoch=epoch, video_id=SubwayVideo.MALL1)
    elif dataset == "mall2":
        protocol = SubwayProtocol(base_log_dir=log_dir, epoch=epoch, video_id=SubwayVideo.MALL2)
    elif dataset == "mall3":
        protocol = SubwayProtocol(base_log_dir=log_dir, epoch=epoch, video_id=SubwayVideo.MALL3)
    elif KitsuneProtocol.is_kitsune_id(dataset):
        protocol = KitsuneProtocol(base_log_dir=log_dir, epoch=epoch, kitsune_dataset=dataset)
    elif dataset == "emoly":
        protocol = EmolyProtocol(base_log_dir=log_dir, epoch=epoch)
    else:
        raise ValueError("Invalid dataset : `{}`".format(dataset))

    if mode == "train":
        protocol.train_model()
    elif mode == "test":
        protocol.test_model()
    else:
        protocol.log_model_latent_codes(config=protocol.get_test_config())


if __name__ == "__main__":
    main()
