from dlssvm_tracker import *
import numpy as np
import time, pickle, sys
from tracker_utils import *

frames, truths = load_dataset("Basketball")
target = np.array(truths[0])

def test(**kwargs):
    print("Testing", kwargs)
    config = {
        "rescale": 0.8,
        "search": 1.3,
        "step": 2,
        "P": 5, "Q": 10,
        "sv_max": 100
    }
    config.update(kwargs)
    tracker = Tracker(frames, target, config)
    t = time.time()
    outputs = [output for _, output in tracker.track()]
    t = time.time() - t
    fps = len(frames) / t
    data = { "fps": fps, "accuracy": accuracy(truths, outputs) }
    print(data)
    data["outputs"] = outputs
    data["config"] = config
    return data

def save(name, data):
    print("Saving", name)
    with open("test1/" + name, "wb") as f:
        pickle.dump(data, f)

test_rescale = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_search = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
test_step = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test_P = [1, 2, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]
test_Q = [1, 2, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100]
test_sv_max = map(lambda x: x+5, range(0, 200, 5))

save("sv_max.pkl", [test(sv_max=i) for i in test_sv_max])
save("search.pkl", [test(search=i) for i in test_search])
save("step.pkl", [test(step=i) for i in test_step])
save("rescale.pkl", [test(rescale=i) for i in test_rescale])
save("P.pkl", [test(P=i) for i in test_P])
save("Q.pkl", [test(Q=i) for i in test_Q])