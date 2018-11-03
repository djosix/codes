"""Display the result created by 'test_tracker.py' or 'test_realtime_tracker.py'.

Usage:

        $ python display_result.py <dataset_name>

    This will load frames in './datasets/<dataset_name>/img',
    and the result is stored in <dataset_name>_result.pkl

Example:

        $ python display_result.py Basketball

    This will load frames in './datasets/Basketball/img',
    and the result is stored in Basketball_result.pkl
"""

import matplotlib.pyplot as plt
import pickle, os, sys
from tracker_utils import load_dataset, draw_rect_on_frame

name = sys.argv[1].rstrip("/")
result_file = name + "_result.pkl"


with open(result_file, "rb") as f:
    result = pickle.load(f)
    outputs = result["outputs"]
    fps = result["fps"]

print("fps:", fps)

frames, truths = load_dataset(name) 

im = plt.imshow(frames[0])
for frame, output, truth in zip(frames, outputs, truths):
    frame = draw_rect_on_frame(frame, output, color=[0, 255, 0])
    frame = draw_rect_on_frame(frame, truth, color=[255, 0, 0])
    im.set_data(frame)
    plt.pause(0.001)
