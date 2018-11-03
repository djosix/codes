"""DLSSVM optimizer."""

import numpy as np
import math

from .pattern import Pattern, SupportVector

thr = 0.0001

class Optimizer(object):

    def __init__(self, config):
        """Initialize Optimizer."""
        self.frame_id = 0
        self.patterns = {}
        self.w = []
        self.P = config["P"]
        self.Q = config["Q"]
        self.sv_max = config["sv_max"]


    def fit(self, frame_id, sampler):
        """Add pattern and optimize classifier."""
        self.frame_id = frame_id
        self.sampler = sampler
        self._add_pattern()

        for p in range(self.P):
            n = len(self.patterns)
            key_id = math.floor(p * n / self.P)
            keys = list(self.patterns.keys())
            id = keys[(-key_id-1) % n]

            self._update_working_set(id)
            self._update_alpha(id)
            self._maintain_sv_number()

            for q in range(self.Q):
                n = len(self.patterns)
                key_id = math.floor(q * n / self.Q)
                keys = list(self.patterns.keys())
                id = keys[(-key_id-1) % n]

                self._update_alpha(id)


    def predict(self, sampler):
        """Predict output from the samples."""
        (X, Y) = sampler.samples
        score = np.inner(self.w, X)
        max_i = np.argmax(score)
        return Y[max_i]


    def _add_pattern(self):
        """Get samples from sampler and add pattern."""
        x_i = self.sampler.target_feat.reshape(1, -1)
        y_i = self.sampler.target.reshape(1, -1)
        (X, Y) = self.sampler.samples
        dX = x_i - X
        
        self.patterns[self.frame_id] = Pattern(x_i, y_i, dX, Y)
        if not len(self.w):
            self.w = np.zeros(x_i.shape)
    

    def _update_working_set(self, pat_id):
        """Let the most violating sample in a pettern to be
        a support vector."""
        pat = self.patterns[pat_id]
        score = pat.loss - self.w.dot(pat.dX.T)
        max_i = np.argmax(score)
        if max_i not in pat.sv:
            self.patterns[pat_id].sv[max_i] = SupportVector()


    def _update_alpha(self, pat_id):
        """Optimize w and alpha."""
        pat = self.patterns[pat_id]
        if len(pat.sv):

            sv_keys = list(pat.sv.keys())
            score = 0
            if len(self.w):
                loss = pat.loss[sv_keys]
                dX = pat.dX[sv_keys]
                score = loss - self.w.dot(dX.T)

            order = np.argsort(score).ravel()
            sv_order = np.array(sv_keys)[order]
            pat.sv_order = sv_order

            n = pat.sv_order[-1]
            g_ij = pat.loss[n] - self.w.dot(pat.dX[n])
            h_ij = pat.dX[n].dot(pat.dX[n].T) + 10e-8

            alpha = pat.sv[n].alpha
            alpha_i = sum([pat.sv[j].alpha for j in pat.sv])
            alpha_star = min(max(-alpha_i, g_ij / h_ij), 1 - alpha_i)

            self.w = self.w + alpha_star * pat.dX[n]
            alpha = alpha + alpha_star
            weight = alpha * alpha * h_ij
            
            if alpha:
                pat.alpha = alpha
                pat.sv[n].weight = weight
            else:
                self._remove_sv(pat_id, n)


    def _remove_sv(self, pat_id, sv_id):
        """Remove support vector."""
        pat = self.patterns[pat_id]
        sv_num = len(pat.sv)
        if sv_num < 2 and pat_id > 0:
            del pat
        else:
            del pat.sv[sv_id]


    def _maintain_sv_number(self):
        """Control the number of all support vectors."""
        while self._get_sv_number() > self.sv_max:

            (p_id, sv_id, min_weight) = (0, 0, np.inf)
            for p in self.patterns:

                pat = self.patterns[p]
                for i in pat.sv:
                    if p == 0 and i == pat.sv_order[0]:
                        continue
                    weight = pat.sv[i].weight
                    if weight < min_weight:
                        min_weight = weight
                        (p_id, sv_id) = (p, i)
            
            alpha = self.patterns[p_id].sv[sv_id].alpha
            self.w = self.w - alpha * self.patterns[p_id].dX[sv_id]

            if len(self.patterns[p_id].sv) < 2:
                del self.patterns[p_id]
            else:
                del self.patterns[p_id].sv[sv_id]


    def _get_sv_number(self):
        """Return the number of all support vectors."""
        return sum([len(self.patterns[p].sv) for p in self.patterns])
        