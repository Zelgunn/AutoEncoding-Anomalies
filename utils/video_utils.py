import numpy as np
import cv2


def show_optical_flow(video_filepath: str):
    video_capture = cv2.VideoCapture(video_filepath)

    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    hsv = np.zeros(shape=[frame_height, frame_width], dtype=np.float32)
    hsv = np.repeat(hsv[:, :, np.newaxis], 3, axis=2)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 40880)

    ret, current_frame = video_capture.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    while ret:
        prev_frame = current_frame
        ret, current_frame = get_next_frame(video_capture, skip=0, output_mean=True)

        flow = cv2.calcOpticalFlowFarneback(prev=prev_frame, next=current_frame, flow=None,
                                            pyr_scale=0.5, levels=25, winsize=8,
                                            iterations=4, poly_n=5, poly_sigma=1.2,
                                            flags=0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 0] = ang * 180 / np.pi
        hsv[..., 1] = 1
        # TODO : Normalize at the end, not during
        hsv[..., 2] = mag
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        _, hsv = cv2.threshold(hsv, thresh=0.5, maxval=255.0, type=cv2.THRESH_TOZERO)

        cv2.imshow("Frame", current_frame)
        cv2.imshow("Flow", hsv)
        cv2.waitKey(30)


def get_next_frame(video_capture, skip=0, output_mean=True):
    ret, frame = video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames = [frame]

    for i in range(skip):
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(frame)

    frames = np.stack(frames, axis=0)

    if output_mean:
        frames = frames.astype(dtype=np.float32)
        frames = frames.mean(axis=0)
        frames = frames.astype(dtype=np.uint8)
        return ret, frames
    else:
        return ret, frames[-1]


show_optical_flow("../datasets/subway/exit/Subway_Exit_192x256.avi")
