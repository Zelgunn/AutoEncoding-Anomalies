from tensorflow.python.keras.saving.saving_utils import model_input_signature

from protocols.audio_video_protocols import AudiosetProtocol


def main():
    protocol = AudiosetProtocol(base_log_dir="../logs/AEA", epoch=73)
    protocol.load_weights(73)
    model = protocol.model

    print("call.input_signature", model.call.input_signature)
    print("model_input_signature", model_input_signature(model))
    print("_get_save_spec(dynamic_batch=True)", model._get_save_spec(dynamic_batch=True))
    print("_saved_model_inputs_spec", model._saved_model_inputs_spec)

    print("Test ?")
    model.save("../logs/AEA/audio_video/audioset/pretrained_modal_sync_73")


if __name__ == "__main__":
    main()
