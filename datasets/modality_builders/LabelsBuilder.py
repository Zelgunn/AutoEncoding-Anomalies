import numpy as np
from typing import Union, List, Iterator, Tuple

from datasets.modality_builders import VideoReader


class LabelsBuilder(object):
    def __init__(self,
                 labels_source: Union[str, np.ndarray, List[str], bool, int, float, List[Tuple[int, int]]],
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

        elif isinstance(labels_source, list):
            assert all([hasattr(timestamp, "__len__") for timestamp in labels_source])
            assert all([len(timestamp) == 2 for timestamp in labels_source])
            labels_source = np.array(labels_source)

        self.labels_source = labels_source
        self.shard_count = shard_count
        self.shard_size = shard_size

    def __iter__(self):
        if isinstance(self.labels_source, np.ndarray):
            if self.labels_source.ndim == 0:
                return self.yield_sample_labels()
        elif isinstance(self.labels_source, VideoReader):
            return self.yield_video_reader_labels()
        else:
            # TODO : Implement for frame labels (that are not videos)
            # TODO : Implement for timestamps
            assert isinstance(self.labels_source, np.ndarray), \
                "Labels source type not expected, received : {}".format(type(self.labels_source))
            return self.yield_timestamps_labels()

    def yield_sample_labels(self):
        labels = [0.0, 1.0] if self.labels_source else [0.0, 0.0]
        for i in range(self.shard_count):
            yield labels

    # region yield frame labels
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
                    labels += [start, position]

            previous_state = state
            i += 1
            if i == self.shard_size:
                if state:
                    labels += [start, 1.0]
                # elif len(labels) == 0:
                #    labels.append([0.0, 0.0])
                # -> use if we can't place empty lists in tfrecords
                i = 0
                yield labels

        if i != 0:
            if previous_state:
                labels += [start, 1.0]
            yield labels

    def yield_video_reader_state(self):
        for frame in self.labels_source:
            state = np.any(frame.astype(np.float) > 0.5)
            yield state

    def yield_video_reader_labels(self):
        return self.yield_frame_labels(self.yield_video_reader_state())

    # endregion

    def yield_timestamps_labels(self):
        for i in range(self.shard_count):
            labels = []
            shard_start = i * self.shard_size
            shard_end = shard_start + self.shard_size - 1
            for start, end in self.labels_source:
                start_in = end > shard_start > start
                end_in = end > shard_end > start

                if start_in and end_in:
                    labels = [(0.0, 1.0)]
                    break

                timestamps_in_shard = (shard_end > start > shard_start and shard_end > end > shard_start)
                if start_in or end_in or timestamps_in_shard:

                    if start_in:
                        label_start = 0.0
                    else:
                        label_start = inverse_lerp(shard_start, shard_end, start)

                    if end_in:
                        label_end = 1.0
                    else:
                        label_end = inverse_lerp(shard_start, shard_end, end)

                    labels.append((label_start, label_end))

            yield labels


def inverse_lerp(start, end, x):
    return (x - start) / (end - start)

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
