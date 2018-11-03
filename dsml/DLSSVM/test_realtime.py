"""
Test real-time tracker

Usage:

        $ python test_realtime.py <dataset_name>

    This will load frames in './datasets/<dataset_name>/img',
    and the result is stored in <dataset_name>_result.pkl

Example:

        $ python test_realtime.py Basketball

    This will load frames in './datasets/Basketball/img',
    and the result is stored in Basketball_result.pkl
"""

from dlssvm_tracker import *
import numpy as np, time, os, pickle, sys
from tracker_utils import load_dataset, draw_rect_on_frame

name = sys.argv[1]
(frames, truths) = load_dataset(name)

# Initialize tracker
config = {
    "rescale": 0.5,
    "search": 1.5,
    "step": 1,
    "P": 50, "Q": 5,
    "sv_max": 50
}

print("tracker.config:", config)

tracker = RealtimeTracker(config)
target = np.array(truths[0])
tracker.set_target(target)
outputs = []


# Start tracking
print("Start real-time tracking")
print("---------------------------------------")

fps_sum, frame_sum = 0, 0

for frame in frames:

    t = time.time()
    output = tracker.track(frame)
    tt = time.time() - t

    # Show tracker status
    patterns = tracker.optimizer.patterns
    sv_count, pat_count = 0, 0
    for p in patterns:
        keys = list(patterns[p].sv.keys())
        pat_count += 1
        sv_count += len(keys)
        print(p, keys)

    # Calculate FPS
    fps = 1 / tt
    fps_sum += fps
    frame_sum += 1

    print("Frame: %d  FPS: %-5.2f  pat: %d  sv: %d  time: %-.2f"
            % (tracker.frame_id, fps, pat_count, sv_count, tt))
    print("---------------------------------------")

    outputs.append(output)


print("Saving outputs to", name + "_result.pkl")

with open(name + "_result.pkl", "wb") as f:
    data = {"fps": fps_sum/frame_sum, "outputs": outputs}
    pickle.dump(data, f)
