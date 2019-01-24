import numpy as np
import cv2

from scheme import Dataset


class SubwayDataset(Dataset):

    def load(self, dataset_path: str, **kwargs):
        video_capture = cv2.VideoCapture(dataset_path)

        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 1000
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(frame_count, frame_height, frame_width)

        video = np.empty(shape=[frame_count, frame_height, frame_width, 3])

        i, ret = 0, True
        while (i < frame_count) and ret:
            ret, video[i] = video_capture.read()
            i += 1

        video_capture.release()


subway_dataset = SubwayDataset(dataset_path="/home/zelgunn/Downloads/events/subway_exit_turnstiles.AVI")
