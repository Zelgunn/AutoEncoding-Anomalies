import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

from protocols.video_protocols import UCSDProtocol, AvenueProtocol, SubwayProtocol
from protocols.video_protocols.SubwayProtocol import SubwayVideo
from datasets.data_readers import VideoReader
from misc_utils.math_utils import standardize_from
from custom_tf_models import LED


def main():
    protocol = SubwayProtocol(video_id=SubwayVideo.ENTRANCE)
    subset = protocol.dataset_loader.train_subset
    pattern = protocol.get_train_pattern()
    dataset = subset.make_tf_dataset(pattern, 42, None, 1)

    for batch in dataset:
        video = batch.numpy()[0]
        video = normalize(video)
        video_mean = np.mean(video, axis=0, keepdims=True)
        video_minus_mean = video - video_mean
        video_minus_mean = normalize(video_minus_mean)

        i = 0
        k = 0
        while k not in [13, 27]:
            frame = video_minus_mean[i]
            frame = cv2.resize(frame, (512, 512))
            cv2.imshow("frame", frame)

            frame_base = video[i]
            frame_base = cv2.resize(frame_base, (512, 512))
            cv2.imshow("frame_base", frame_base)

            cv2.imshow("frame_bis", np.abs(frame * 2 - 1))

            k = cv2.waitKey(100)
            i = (i + 1) % len(video)

        if k == 27:
            break


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    main()
