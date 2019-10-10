from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol, SubwayProtocol


def main():
    train = 1
    initial_epoch = 0
    dataset = "ped1"

    if dataset == "ped2":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=2)
    elif dataset == "ped1":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=1)
    elif dataset == "avenue":
        protocol = AvenueProtocol(initial_epoch=initial_epoch)
    elif dataset == "shanghaitech":
        protocol = ShanghaiTechProtocol(initial_epoch=initial_epoch)
    elif dataset == "subway":
        protocol = SubwayProtocol(initial_epoch=initial_epoch)
    else:
        raise ValueError

    # protocol.autoencode_video(r"D:\Users\Degva\Documents\_PhD\Tensorflow\datasets\ucsd\ped1\Test\Test003",
    #                           initial_epoch)

    if train:
        protocol.train_model()
    else:
        protocol.test_model()


if __name__ == "__main__":
    main()
