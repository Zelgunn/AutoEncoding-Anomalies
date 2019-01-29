import numpy as np
import cv2
import os
import time
from tqdm import tqdm

from datasets import Dataset


class SubwayDataset(Dataset):
    def __init__(self, **kwargs):
        self.video_capture = None
        self.frame_count = None
        self.frame_height = None
        self.frame_width = None
        self.fourcc = None
        self.fps = None
        super(SubwayDataset, self).__init__(**kwargs)

    def load(self, dataset_path: str, **kwargs):
        video_filepath = os.path.join(dataset_path, "Subway_Exit.avi")
        self.video_capture = cv2.VideoCapture(video_filepath)

        self.frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fourcc = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

    def resize_video(self, target_size):
        height, width = target_size
        video_filepath = os.path.join(self.dataset_path, "Subway_Exit_{0}x{1}.avi".format(height, width))
        frame_size = (width, height)
        video_writer = cv2.VideoWriter(video_filepath, self.fourcc, self.fps, frame_size)

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.frame_count):
            print("\r{0}/{1}".format(i + 1, self.frame_count), end='')
            ret, frame = self.video_capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, dsize=frame_size)
            video_writer.write(frame)


# subway_dataset = SubwayDataset(dataset_path="/home/zelgunn/Documents/datasets/Subway")
# for n in range(6):
#     subway_dataset.resize_video(target_size=[192 // (2 ** n), 256 // (2 ** n)])
