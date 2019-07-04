import tensorflow as tf
import numpy as np
import dlib
import cv2
from typing import Dict, Union, List, Any

from modalities import Modality


class Landmarks(Modality):
    def __init__(self, dlib_shape_predictor_path: str):
        super(Landmarks, self).__init__()

        self.dlib_shape_predictor_path = dlib_shape_predictor_path

        self.dlib_face_detector = dlib.get_frontal_face_detector()
        self.dlib_shape_predictor = dlib.shape_predictor(dlib_shape_predictor_path)

    def get_config(self) -> Dict[str, Any]:
        base_config = super(Landmarks, self).get_config()
        config = {
            "dlib_shape_predictor_path": self.dlib_shape_predictor_path,
        }
        return {**base_config, **config}

    @classmethod
    def encode_to_tfrecord_feature(cls, modality_value) -> Dict[str, tf.train.Feature]:
        return cls.encode_raw(modality_value, np.float32)

    @classmethod
    def decode_from_tfrecord_feature(cls, parsed_features):
        return cls.decode_raw(parsed_features, tf.float32)

    @classmethod
    def tfrecord_features(cls) -> Dict[str, tuple]:
        return {cls.id(): tf.io.VarLenFeature(tf.string),
                cls.shape_id(): cls.tfrecord_shape_parse_function()}

    @classmethod
    def rank(cls) -> int:
        return 3

    def compute_landmarks(self, frames: np.ndarray, upsampling=(1, 2), use_other_if_fail=False) -> np.ndarray:
        frame_count, height, width, _ = frames.shape

        if frames.dtype != np.uint8:
            if frames.dtype in [np.float32, np.float64]:
                if frames.max() < 1.0:
                    frames = frames * 255
                frames = frames.astype(np.uint8)
            else:
                raise ValueError("Frames must either be uint8, float32 or float64, found {}".format(frames.dtype))

        landmarks = np.empty(shape=[frame_count, 68, 2], dtype=np.float32)

        failed_indexes = []

        for i in range(frame_count):
            frame = frames[i]

            bounding_box = []

            for upsampling_value in upsampling:
                bounding_box = self.dlib_face_detector(frame, upsampling_value)
                if len(bounding_box) != 0:
                    break

            if len(bounding_box) == 0:
                if use_other_if_fail:
                    failed_indexes.append(i)
                else:
                    raise ValueError("Could not find a face in image.")
            else:
                bounding_box = bounding_box[0]

                shape = self.dlib_shape_predictor(frame, bounding_box)

                for j in range(68):
                    landmark = shape.part(j)
                    landmarks[i, j, :] = [landmark.x, landmark.y]

        for i in failed_indexes:
            copy_index = None
            for j in range(0, frame_count):
                if j not in failed_indexes:
                    if copy_index is None:
                        copy_index = j
                    elif abs(copy_index - i) > abs(j - i):
                        copy_index = j
            if copy_index is None:
                raise ValueError("Could not find any face in this video.")
            landmarks[i] = landmarks[copy_index]

        if use_other_if_fail:
            print(" - Failed {} detections.".format(len(failed_indexes)))

        landmarks /= [width, height]

        if use_other_if_fail:
            print(" - Failed to identify landmarks {} times".format(len(failed_indexes)))

        return landmarks


def display_dlib_landmarks_on_video(dlib_shape_predictor_path: str,
                                    video: Union[str, cv2.VideoCapture, np.ndarray, List[str]],
                                    buffer_size=25,
                                    display_size=None):
    from datasets.data_readers import VideoReader
    video_reader = VideoReader(video)
    if buffer_size is None:
        buffer_size = video_reader.frame_count
    buffer = np.empty([buffer_size, *video_reader.frame_shape], dtype=np.uint8)
    height, width = video_reader.frame_size if display_size is None else display_size

    landmarks_modality = Landmarks(dlib_shape_predictor_path)

    i = 0
    for frame in video_reader:
        buffer[i] = frame
        i += 1
        if i == buffer_size:
            landmarks = landmarks_modality.compute_landmarks(buffer)
            landmarks = landmarks * [width, height]
            for frame_index in range(buffer_size):
                frame = buffer[frame_index]
                if display_size is not None:
                    frame = cv2.resize(frame, (display_size[1], display_size[0]))

                for landmark_index in range(68):
                    point = landmarks[frame_index][landmark_index]
                    point = (int(point[0]), int(point[1]))
                    cv2.drawMarker(frame, point, (0, 0, 255))
                cv2.imshow("frame", frame)
                cv2.waitKey(30)
            i = 0
