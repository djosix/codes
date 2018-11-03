"""To keep support vectors and seen patterns."""

import numpy as np

class Pattern(object):
    def __init__(self, x_i, y_i, dX, Y):
        self.x_i = np.array(x_i)
        self.y_i = np.array(y_i)
        self.dX = np.array(dX)
        self.Y = np.array(Y)
        self.sv = {}
        self.sv_order = []
        self.loss = self._compute_loss()
    
    def _compute_loss(self):
        """Compute all loss between y_i and y."""
        (x0, y0, w0, h0) = self.Y.T
        (x, y, w, h) = self.y_i.reshape(4, 1).repeat(self.Y.shape[0], axis=1)
        (x0_, y0_, x_, y_) = (x0+w0, y0+h0, x+w, y+h)
        (left, top) = (np.maximum(x, x0), np.maximum(y, y0))
        (right, bottom) = (np.minimum(x_, x0_), np.minimum(y_, y0_))
        intersect = np.maximum(bottom - top, 0) * np.maximum(right - left, 0)
        union = w * h + w0 * h0 - intersect
        return 1 - intersect / union


class SupportVector(object):
    def __init__(self, alpha=10e-8, weight=0):
        self.alpha = alpha
        self.weight = weight
