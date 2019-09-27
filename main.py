from protocols.video_protocols import UCSDProtocol, AvenueProtocol, ShanghaiTechProtocol


def main():
    train = 1
    initial_epoch = 0
    dataset = "shanghaitech"

    if dataset is "ped2":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=2)
    elif dataset is "ped1":
        protocol = UCSDProtocol(initial_epoch=initial_epoch, dataset_version=1)
    elif dataset is "avenue":
        protocol = AvenueProtocol(initial_epoch=initial_epoch)
    elif dataset is "shanghaitech":
        protocol = ShanghaiTechProtocol(initial_epoch=initial_epoch)
    else:
        raise ValueError

    if train:
        protocol.train_model()
    else:
        protocol.test_model()


if __name__ == "__main__":
    main()
