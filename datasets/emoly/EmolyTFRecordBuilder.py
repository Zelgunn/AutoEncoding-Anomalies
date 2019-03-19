import cv2
import os
import json


class EmolyTFRecordBuilder(object):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def build(self, shard_size: int):
        self.build_video_tf_records(shard_size)

    def build_video_tf_records(self, shard_size: int):
        with open(os.path.join(self.videos_folder, "header.json"), 'w') as file:
            json.dump({"shard_size": shard_size}, file)

        for video_filename in self.list_videos_filenames():
            video_filepath = os.path.join(self.videos_folder, video_filename)

            video_target_folder = video_filename[:-4]  # remove mp4
            video_target_folder = os.path.join(self.videos_folder, video_target_folder)

            if not os.path.exists(video_target_folder):
                os.mkdir(video_target_folder)
            else:
                assert os.path.isdir(video_target_folder)

            video_capture = cv2.VideoCapture(video_filepath)

            video_capture.release()

    def list_videos_filenames(self):
        elements = os.listdir(self.videos_folder)
        videos_filenames = []
        for video_filename in elements:
            if ".mp4" in video_filename and os.path.isfile(os.path.join(self.videos_folder, video_filename)):
                videos_filenames.append(video_filename)
        return videos_filenames

    @property
    def videos_folder(self):
        return os.path.join(self.dataset_path, "video")


emoly_tf_record_builder = EmolyTFRecordBuilder("../datasets/emoly")
emoly_tf_record_builder.build(shard_size=32)

# def encode():
#     ucsd_filepath = r"..\datasets\ucsd\ped2\Train.npz"
#     video = np.load(ucsd_filepath)["videos"]
#     video = np.concatenate(video) * 255
#
#     batch_count = np.ceil(len(video) / SHARD_SIZE).astype(np.int64)
#
#     for i in range(batch_count):
#         filepath = train_filepath.format("2_{}".format(i))
#         with tf.python_io.TFRecordWriter(filepath) as writer:
#             features = convert_function(video[i * SHARD_SIZE:(i + 1) * SHARD_SIZE])
#
#             tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
#             writer.write(tfrecord_example.SerializeToString())
