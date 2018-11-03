"""
Test real-time tracking on camera

Usage:
        $ python test_camera.py

    You should compile OpenCV for Python3
    before running this script.
"""

from dlssvm_tracker import *
import numpy as np, time, cv2
from tracker_utils import draw_rect_on_frame


# Configure tracker
config = {
    "rescale": 0.2,
    "search": 1.3,
    "step": 2,
    "P": 5, "Q": 10,
    "sv_max": 100
}

print("tracker.config:", config)

tracker = RealtimeTracker(config)
init_rect = np.array([120, 80, 100, 100])
tracker.set_target(init_rect)


# Add mouse event to reset tracking target
is_cropping = False
def reset_tracker(event, x, y, flags, param):
    global tracker, init_rect, is_cropping
    if event == cv2.EVENT_LBUTTONDOWN and not is_cropping:
        is_cropping = True
        init_rect[0:2] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP and is_cropping:
        is_cropping = False
        init_rect[2:4] = [x, y]
        init_rect[0::2].sort()
        init_rect[1::2].sort()
        init_rect[2:4] -= init_rect[0:2]
        if all(init_rect[2:4] > [5, 5]):
            print("Reset target:", init_rect)
            tracker.set_target(init_rect)
    
cv2.namedWindow("tracker")
cv2.setMouseCallback("tracker", reset_tracker)

# Start captureing video from camera
cap = cv2.VideoCapture(0)

# Start tracking
print("Start real-time tracking on camera")
print("---------------------------------------")
try:
    while True:
        (_, frame) = cap.read()

        frame = np.fliplr(frame) # mirror

        # track loop output
        t = time.time()
        output = tracker.track(frame)
        tt = time.time() - t
        fps = 1 / tt

        # Show tracker status
        patterns = tracker.optimizer.patterns
        sv_count, pat_count = 0, 0
        for p in patterns:
            keys = list(patterns[p].sv.keys())
            pat_count += 1
            sv_count += len(keys)
            print(p, keys)
        
        # Print parameters
        print("Frame: %d  FPS: %-5.2f  pat: %d  sv: %d  time: %-.2f"
                % (tracker.frame_id, fps, pat_count, sv_count, tt))
        print("---------------------------------------")
        
        # Draw tracking box
        frame = draw_rect_on_frame(frame, output, 10, [0, 255, 0])

        # Show frame
        cv2.imshow("tracker", frame)
        cv2.waitKey(1)

except ValueError:
    cap.release()
    cv2.destroyAllWindows()
