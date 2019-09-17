import tensorflow as tf
import numpy as np
import dlib
import cv2
from typing import Dict, Tuple

from modalities import RawVideo
from modalities.RawVideo import video_feature


class Faces(RawVideo):
    def __init__(self):
        super(Faces, self).__init__()

        self.dlib_face_detector = dlib.get_frontal_face_detector()

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        if modality_value.ndim == 3:
            modality_value = np.expand_dims(modality_value, axis=-1)

        return {cls.id(): video_feature(modality_value)}

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features: Dict[str, tf.SparseTensor]):
        encoded_raw_video: tf.Tensor = parsed_features[cls.id()].values

        raw_video_shard_size = tf.shape(encoded_raw_video)[0]
        raw_video = tf.map_fn(lambda i: tf.cast(tf.image.decode_jpeg(encoded_raw_video[i]), tf.float32),
                              tf.range(raw_video_shard_size),
                              dtype=tf.float32)

        return raw_video

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.id(): tf.io.VarLenFeature(tf.string)}

    @classmethod
    def rank(cls) -> int:
        return 4

    def compute_faces(self,
                      frames: np.ndarray,
                      frame_size: Tuple[int, int]
                      ) -> np.ndarray:
        bounding_box = self.get_video_bounding_box(frames)
        print(bounding_box)
        exit()

        frame_count = len(frames)
        channels = 1 if frames.ndim == 3 else frames.shape[-1]
        faces = np.empty(shape=(frame_count, *frame_size, channels), dtype=np.float32)

        for i in range(frame_count):
            face = frames[i][y:y + j, x:x + w]
            face = RawVideo.compute_raw_frame(face, frame_size)
            faces[i] = face

        return faces

    def compute_face(self,
                     frame: np.ndarray,
                     frame_size: Tuple[int, int]
                     ) -> np.ndarray:

        for upsampling_value in (1, 2):
            bounding_box = self.dlib_face_detector(frame, upsampling_value)
            if len(bounding_box) != 0:
                break

        bounding_box = bounding_box[0]
        print(bounding_box)
        exit()
        face = frame[y:y + j, x:x + w]

        return RawVideo.compute_raw_frame(face, frame_size)

    def get_video_bounding_box(self, frames: np.ndarray):
        bounding_box = None
        for i in range(len(frames)):
            bounding_box = self.get_frame_bounding_box(frames[i])
            print(bounding_box)
            exit()

        return bounding_box

    def get_frame_bounding_box(self, frame: np.ndarray):
        for upsampling_value in (1, 2):
            bounding_box = self.dlib_face_detector(frame, upsampling_value)
            if len(bounding_box) != 0:
                break
        bounding_box = bounding_box[0]
        return bounding_box
