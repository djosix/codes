from dlssvm_tracker import *
import numpy as np
import time, pickle, sys
from tracker_utils import *
import sqlite3

frames, truths = load_dataset("Basketball")
target = np.array(truths[0])

def test(config):
    print("Testing", config)
    tracker = Tracker(frames, target, config)
    t = time.time()
    outputs = [output for _, output in tracker.track()]
    t = time.time() - t
    fps = len(frames) / t
    data = { "fps": fps, "accuracy": accuracy(truths, outputs) }
    print(data)
    # data["outputs"] = outputs
    return data

def save(name, data):
    print("Saving", name)
    with open(name, "wb") as f:
        pickle.dump(data, f)

test_rescale = [0.5, 0.6, 0.7]
test_step = [1, 2, 3]
test_p = [5, 10, 20, 50]
test_q = [5, 10, 20]
test_search = [1.3, 1.5]
test_svmax = [20, 50, 100]

configs = []
for scale in test_rescale:
    for step in test_step:
        for p in test_p:
            for q in test_q:
                for r in test_search:
                    for v in test_svmax:
                        configs.append({
                            "rescale": scale,
                            "search": r,
                            "step": step,
                            "P": p, "Q": q,
                            "sv_max": v
                        })

conn = sqlite3.connect("results.db")
c = conn.cursor()
for i, config in enumerate(configs):
    print(i, "/", len(configs))
    result = test(config)
    c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        i,
        config["rescale"],
        config["step"],
        config["search"],
        config["P"],
        config["Q"],
        config["sv_max"],
        result["fps"],
        result["accuracy"]
    ))
    conn.commit()
conn.close()



