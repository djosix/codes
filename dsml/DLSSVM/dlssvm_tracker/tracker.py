"""Trackers."""


import os, pickle
import numpy as np
from skimage.transform import rescale

from .optimizer import Optimizer
from .sampler import Sampler


def rescale_frame(frame, scale):
    if scale == 1:
        return frame
    return rescale(frame, scale)

def rescale_rect(rect, scale):
    if scale == 1:
        return rect
    return (rect * scale).astype(int)

def restore_rect(rect, scale):
    if scale == 1:
        return rect
    rect = rect + [0, 0, 1, 1]
    return (rect / scale).astype(int)


DEFAULT_CONFIG = {
    "rescale": 0.8,
    "search": 1.3,
    "step": 2,
    "P": 5, "Q": 10,
    "sv_max": 100,
}


class Tracker(object):
    def __init__(self, frames, rect, config=None):
        """Track a target form frames."""
        self.frame_id = 0
        self.frames = frames
        self.config = config or DEFAULT_CONFIG
        self.scale = self.config["rescale"]
        self.target = rescale_rect(rect, self.scale)

    def track(self):
        """Generate results."""
        self.optimizer = Optimizer(self.config)
        for frame in self.frames:
            scaled = rescale_frame(frame, self.scale)
            sampler = Sampler(scaled, self.target, self.config)
            if self.frame_id > 0:
                self.target = self.optimizer.predict(sampler)
                sampler = Sampler(scaled, self.target, self.config)
            self.optimizer.fit(self.frame_id, sampler)
            yield (frame, restore_rect(self.target, self.scale))
            self.frame_id += 1
        self.frame_id = 0


class RealtimeTracker(object):
    def __init__(self, config):
        self.config = config or DEFAULT_CONFIG
        self.scale = self.config["rescale"]
    
    def set_target(self, target):
        """Set target and initialize optimizer."""
        self.frame_id = 0
        self.target = rescale_rect(target, self.scale)
        self.optimizer = Optimizer(self.config)

    def track(self, frame):
        """Output result."""
        frame = rescale_frame(frame, self.scale)
        sampler = Sampler(frame, self.target, self.config)
        if self.frame_id > 0:
            self.target = self.optimizer.predict(sampler)
            sampler = Sampler(frame, self.target, self.config)
        self.optimizer.fit(self.frame_id, sampler)
        self.frame_id += 1
        return restore_rect(self.target, self.scale)
