from datasets.tfrecord_builders import TFRecordBuilder


class SubwayTFRecordBuilder(TFRecordBuilder):
    def build(self, shard_size):
        pass


if __name__ == "__main__":
    subway_tf_record_builder = SubwayTFRecordBuilder("../datasets/subway")
    subway_tf_record_builder.build(shard_size=32)
