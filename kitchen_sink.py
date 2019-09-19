from z_kitchen_sink.video_cnn_transformer import train_video_cnn_transformer, test_video_cnn_transformer


def main():
    dataset_name = "emoly"
    channels = 1
    autoencoder_mode = "iae"
    use_transformer = False
    code_size = 256
    input_length, output_length, time_step = 4, 4, 8
    height = width = 64
    initial_epoch = 0
    train = 1
    train_only_embeddings = False and use_transformer
    if train:
        train_video_cnn_transformer(input_length=input_length, output_length=output_length, time_step=time_step,
                                    height=height, width=width,
                                    use_transformer=use_transformer, autoencoder_mode=autoencoder_mode,
                                    train_only_embeddings=train_only_embeddings, copy_regularization_factor=0.1,
                                    initial_epoch=initial_epoch, batch_size=8,
                                    dataset_name=dataset_name, channels=channels, code_size=code_size)
    else:
        test_video_cnn_transformer(input_length=input_length, output_length=output_length, time_step=time_step,
                                   height=height, width=width,
                                   use_transformer=use_transformer, autoencoder_mode=autoencoder_mode,
                                   initial_epoch=initial_epoch,
                                   dataset_name=dataset_name, channels=channels, code_size=code_size)


if __name__ == "__main__":
    main()
