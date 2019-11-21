from datasets.tfrecord_builders.SubwayTFRecordBuilder import SubwayVideo
from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol, SubwayProtocol


def main():
    best_weights = {
        "ped2": 16,
        "ped1": 50,
        "avenue": 22,
        "shanghaitech": 14,
        "exit": 11,
        "entrance": 97,
    }

    train = 0
    initial_epoch = None
    dataset = "ped2"

    if initial_epoch is None:
        if train:
            initial_epoch = 0
        else:
            initial_epoch = best_weights[dataset]

    if dataset == "ped2":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=2)
    elif dataset == "ped1":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=1)
    elif dataset == "avenue":
        protocol = AvenueProtocol(initial_epoch=initial_epoch)
    elif dataset == "shanghaitech":
        protocol = ShanghaiTechProtocol(initial_epoch=initial_epoch)
    elif dataset == "exit":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.EXIT)
    elif dataset == "entrance":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.ENTRANCE)
    elif dataset == "mall1":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.MALL1)
    elif dataset == "mall2":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.MALL2)
    elif dataset == "mall3":
        protocol = SubwayProtocol(initial_epoch=initial_epoch, video_id=SubwayVideo.MALL3)
    else:
        raise ValueError

    # root = r"D:\Users\Degva\Documents\_PhD\Tensorflow\datasets\subway"
    #
    # input_video = root + r"\exit\Subway_Exit.avi"
    # output_video = root + r"\subway_exit.avi"
    #
    # protocol.autoencode_video(video_source=input_video,
    #                           target_path=output_video,
    #                           load_epoch=initial_epoch,
    #                           fps=25.0)
    #
    # exit()

    if train:
        protocol.train_model()
    else:
        protocol.test_model()


if __name__ == "__main__":
    main()
