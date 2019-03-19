import os


class EmolyTFRecordBuilder(object):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def list_videos_filenames(self):
        return os.listdir(self.videos_folder)

    @property
    def videos_folder(self):
        return os.path.join(self.dataset_path, "video")

# emoly_tf_record_builder = EmolyTFRecordBuilder("../datasets/emoly")
# print(emoly_tf_record_builder.list_videos_filenames())
