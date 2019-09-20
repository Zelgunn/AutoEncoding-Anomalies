import tensorflow as tf
import numpy as np
import dlib
from typing import Tuple, Union, Dict

from modalities import Modality, RawVideo


class Faces(Modality):
    def __init__(self):
        super(Faces, self).__init__()

        self.dlib_face_detector = dlib.get_frontal_face_detector()

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
        return 2

    def compute_faces(self,
                      frames: np.ndarray,
                      compute_bounding_box_per_face=True,
                      ) -> np.ndarray:
        length, height, width, _ = frames.shape
        bounding_boxes = np.zeros(shape=(length, 4), dtype=np.float32)
        previous_bounding_box = None
        upsampling_values = (1, 2, 3)
        for i in range(length):
            bounding_box = self.get_frame_bounding_box(frames[i], upsampling_values)
            
            if bounding_box is None:
                if previous_bounding_box is None:
                    previous_bounding_box = self.get_video_bounding_box(frames, upsampling_values)
                    if None in previous_bounding_box:
                        print(" Warning : Could not detect a face on this video.")
                        bounding_boxes[:] = (0, height, 0, width)
                        break

                bounding_box = previous_bounding_box
            bounding_boxes[i] = bounding_box
            previous_bounding_box = bounding_box
            upsampling_values = (1, )

        if not compute_bounding_box_per_face:
            mins = bounding_boxes.min(axis=0)
            maxs = bounding_boxes.max(axis=0)
            bounding_boxes[:] = (mins[0], maxs[1], mins[2], maxs[3])

        bounding_boxes /= (height, height, width, width)
        return bounding_boxes

    def extract_faces(self,
                      frames: np.ndarray,
                      frame_size: Tuple[int, int],
                      compute_bounding_box_per_face=True,
                      ) -> np.ndarray:
        bounding_box = None
        if not compute_bounding_box_per_face:
            bounding_box = self.get_video_bounding_box(frames)

        frame_count = len(frames)
        channels = 1 if frames.ndim == 3 else frames.shape[-1]
        faces = np.empty(shape=(frame_count, *frame_size, channels), dtype=np.float32)

        for i in range(frame_count):
            if compute_bounding_box_per_face:
                frame_bounding_box = self.get_frame_bounding_box(frames[i])
                if frame_bounding_box is not None:
                    bounding_box = frame_bounding_box
                if bounding_box is None:
                    bounding_box = self.get_video_bounding_box(frames)

            min_y, max_y, min_x, max_x = bounding_box

            face = frames[i][min_y:max_y, min_x: max_x]
            face = RawVideo.compute_raw_frame(face, frame_size)
            faces[i] = face

        return faces

    def extract_face(self,
                     frame: np.ndarray,
                     frame_size: Tuple[int, int]
                     ) -> np.ndarray:

        min_y, max_y, min_x, max_x = self.get_frame_bounding_box(frame)
        face = frame[min_y:max_y, min_x: max_x]

        return RawVideo.compute_raw_frame(face, frame_size)

    def get_video_bounding_box(self,
                               frames: np.ndarray,
                               upsampling_values=(1, 2, 3),
                               ) -> Tuple[int, int, int, int]:
        frames = self.convert_frames_for_detector(frames)

        min_y, max_y, min_x, max_x = None, None, None, None
        for i in range(len(frames)):
            frame_bounding_box = self.get_frame_bounding_box(frames[i], upsampling_values)
            if frame_bounding_box is not None:
                if min_y is None:
                    min_y, max_y, min_x, max_x = frame_bounding_box
                else:
                    min_y = min(min_y, frame_bounding_box[0])
                    max_y = max(max_y, frame_bounding_box[1])
                    min_x = min(min_x, frame_bounding_box[2])
                    max_x = max(max_x, frame_bounding_box[3])

        return min_y, max_y, min_x, max_x

    def get_frame_bounding_box(self,
                               frame: np.ndarray,
                               upsampling_values=(1, 2, 3),
                               ) -> Union[Tuple[int, int, int, int], None]:
        frame = self.convert_frames_for_detector(frame)

        bounding_box = ()
        for upsampling_value in upsampling_values:
            bounding_box = self.dlib_face_detector(frame, upsampling_value)
            if len(bounding_box) != 0:
                break

        if len(bounding_box) == 0:
            return None

        bounding_box = bounding_box[0]
        bounding_box = (bounding_box.top(), bounding_box.bottom(), bounding_box.left(), bounding_box.right())
        return bounding_box

    @staticmethod
    def convert_frames_for_detector(frames: np.ndarray) -> np.ndarray:
        if frames.dtype != np.uint8:
            if frames.dtype in [np.float32, np.float64]:
                if frames.max() < 1.0:
                    frames = frames * 255
                frames = frames.astype(np.uint8)
        return frames
