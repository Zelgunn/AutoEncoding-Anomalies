import tensorflow as tf

from protocols.video_protocols import UCSDProtocol


def main():
    protocol = UCSDProtocol(initial_epoch=0, dataset_version=2)

    model = protocol.make_model()

    for layer in model.layers:
        print("|====| ", layer)
        if isinstance(layer, tf.keras.models.Model):
            for sub_layer in layer.layers:
                print(sub_layer)
                if hasattr(sub_layer, "kernel_initializer"):
                    print(sub_layer.kernel_initializer)


if __name__ == "__main__":
    main()
