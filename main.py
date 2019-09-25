from protocols import UCSDProtocol


def main():
    train = 1
    initial_epoch = 16

    protocol = UCSDProtocol(initial_epoch=initial_epoch)

    if train:
        protocol.train_model()
    else:
        protocol.test_model()


if __name__ == "__main__":
    main()
