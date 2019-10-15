from datasets.tfrecord_builders.SubwayTFRecordBuilder import SubwayVideo
from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol, SubwayProtocol


def main():
    train = 0
    initial_epoch = 0
    dataset = "entrance"

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

    # protocol.autoencode_video(r"D:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped1\Test\Test008",
    #                           initial_epoch)

    if train:
        protocol.train_model()
    else:
        protocol.test_model()


if __name__ == "__main__":
    main()
