from z_kitchen_sink.video_cnn_transformer import train_video_cnn_transformer, test_video_cnn_transformer


def main():
    train = 1
    dataset_name = "emoly"
    if train:
        train_video_cnn_transformer(use_transformer=False, initial_epoch=0, batch_size=8,
                                    dataset_name=dataset_name, channels=3)
    else:
        test_video_cnn_transformer(use_transformer=True, initial_epoch=48,
                                   dataset_name=dataset_name, channels=3)


if __name__ == "__main__":
    main()
