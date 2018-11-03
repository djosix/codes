"""
Test tracker and generate <dataset_name>_result.pkl

Usage:

        $ python test_tracker.py <dataset_name>

    This will load frames in './datasets/<dataset_name>/img',
    and the result is stored in <dataset_name>_result.pkl

Example:

        $ python test_tracker.py Basketball

    This will load frames in './datasets/Basketball/img',
    and the result is stored in Basketball_result.pkl
"""

from dlssvm_tracker import *
import numpy as np
import time, pickle, sys
from tracker_utils import load_dataset

# Pass dataset name as an argument
name = sys.argv[1].rstrip("/")

(frames, truths) = load_dataset(name)


# Initialize tracker
config = {
    "rescale": 0.8,
    "search": 1.3,
    "step": 2,
    "P": 5, "Q": 10,
    "sv_max": 100,
}

target = np.array(truths[0])
tracker = Tracker(frames, target, config)
outputs = []

print("tracker.config:", config)


print("Start tracking")
print("---------------------------------------")

# Start tracking
fps_sum, frame_sum = 0, 0
t = time.time()

for (frame, output), truth in zip(tracker.track(), truths):
    
    # Show tracker status
    patterns = tracker.optimizer.patterns
    sv_count, pat_count = 0, 0
    for p in patterns:
        keys = list(patterns[p].sv.keys())
        pat_count += 1
        sv_count += len(keys)
        print(p, keys)

    # Calculate FPS
    tt = time.time() - t
    fps = 1 / tt
    fps_sum += fps
    frame_sum += 1

    print("Frame: %d  FPS: %-5.2f  pat: %d  sv: %d  time: %-.2f\n"
        "---------------------------------------"
        % (tracker.frame_id, fps, pat_count, sv_count, tt))
    t = time.time()

    outputs.append(output)


print("Saving outputs to", name + "_result.pkl")

with open(name + "_result.pkl", "wb") as f:
    data = {"fps": fps_sum/frame_sum, "outputs": outputs}
    pickle.dump(data, f)
