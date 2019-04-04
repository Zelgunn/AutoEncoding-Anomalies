import numpy as np
from typing import Union, List, Iterator

from datasets.modality_builders import VideoReader


class LabelsBuilder(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float],
                 shard_count: int = None,
                 shard_size: int = None,
                 ):
        if isinstance(labels_source, str):
            if labels_source.endswith(".npy") or labels_source.endswith(".npz"):
                labels_source = np.load(labels_source, mmap_mode="r")
            else:
                labels_source = VideoReader(labels_source)

        elif isinstance(labels_source, bool) or \
                isinstance(labels_source, int) or \
                isinstance(labels_source, float):
            labels_source = np.array(labels_source).astype(np.bool)

        self.labels_source = labels_source
        self.shard_count = shard_count
        self.shard_size = shard_size

    def __iter__(self):
        if isinstance(self.labels_source, np.ndarray):
            if self.labels_source.ndim == 0:
                return self.yield_sample_labels()
        else:
            assert isinstance(self.labels_source, VideoReader)

        # elif self.labels_source.ndim == 1:

    def yield_sample_labels(self):
        labels = [0.0, 1.0] if self.labels_source else [0.0, 0.0]
        for i in range(self.shard_count):
            yield labels

    def yield_frame_labels(self, state_iterator: Iterator):
        assert self.shard_size is not None
        i = 0
        previous_state = None
        labels = []
        start = None

        for state in state_iterator:
            if i == 0:
                labels = []
                previous_state = False

            if state != previous_state:
                position = i / (self.shard_size - 1.0)
                if state:
                    start = position
                else:
                    labels.append([start, position])

            previous_state = state
            if (i + 1) == self.shard_size:
                if state:
                    labels.append([start, 1.0])
                # elif len(labels) == 0:
                #    labels.append([0.0, 0.0])
                # -> use if we can't place empty lists in tfrecords
                i = 0
                yield labels

    def yield_video_reader_state(self):
        for frame in self.labels_source:
            state = np.any(frame.astype(np.float) > 0.5)
            yield state

    def yield_video_reader_labels(self):
        return self.yield_frame_labels(self.yield_video_reader_state())

# TODO : From numpy file or array
# TODO : From image collection
# TODO : From video (?)
# TODO : From timestamps
# TODO : From a single value

# if numpy array
#   Y - 0D : is single value (bool)
#   N - 1D : frame labels [time] (bool)
#   Y - 2D : timestamps [n_timestamps, 2] (float - [start:end])
#   N - 3D : spatio-temporal labels [time, height, width]
