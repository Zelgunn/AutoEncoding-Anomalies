from z_kitchen_sink.video_cnn_transformer import train_video_cnn_transformer, test_video_cnn_transformer


def main():
    train = True
    if train:
        train_video_cnn_transformer(use_transformer=False, initial_epoch=0, batch_size=8)
    else:
        test_video_cnn_transformer(use_transformer=True, initial_epoch=31)


if __name__ == "__main__":
    main()
