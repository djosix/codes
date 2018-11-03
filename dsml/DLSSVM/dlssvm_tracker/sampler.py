import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rescale


class Sampler(object):

    def __init__(self, frame, target, config):
        """Initialize Sampler

        Args:
            frame -- a numpy array of image.
            target -- target rectangle: [x, y, width, height].
            config -- a dict like: {"search": 1, "step": 1}.
        """
        self._set_default_config(config)
        self.search = config['search']
        self.scales = config['scales']
        self.step = config['step']
        region = self._get_search_region(frame, target)
        cropped = self._crop_region(frame, region)
        self.features = self._get_features(cropped)
        self.target = target
        self.region = region
        self._sample()


    def _sample(self):
        """Sampling x's in the searching region."""
        (x0, y0) = self.region[0:2]
        (h_f, w_f) = self.features.shape[0:2]
        (x, y, w, h) = self.target
        (X, Y) = ([], [])
        for j in range(0, h_f-h, self.step):
            for i in range(0, w_f-w, self.step):
                X += [self.features[j:j+h, i:i+w].ravel()]
                Y += [[i+x0, j+y0, w, h]]
        self.samples = (np.array(X), np.array(Y))
        (x_, y_) = (x - x0, y - y0)
        cropped = self.features[y_:y_+h, x_:x_+w]
        self.target_feat = cropped.ravel()


    def _set_default_config(self, config):
        """Update blanks to default config."""
        config.setdefault("search", 1)
        config.setdefault("scales", [1])
        config.setdefault("step", 1)


    def _get_search_region(self, frame, target):
        """Return calculated searching region."""
        (x, y, w, h) = target
        r = round(self.search * np.linalg.norm([w, h]) / 2)
        region = np.array([x+w/2-r, y+h/2-r, 2*r, 2*r], dtype=int)
        if region[0] < 0:
            region[0] = 0
        elif region[0] + region[2] > frame.shape[1]:
            region[0] = frame.shape[1] - region[2]
        if region[1] < 0:
            region[1] = 0
        elif region[1] + region[3] > frame.shape[0]:
            region[1] = frame.shape[0] - region[3]
        return region


    def _crop_region(self, frame, region):
        """Return the pixels of the region on the frame."""
        (x, y, w, h) = region
        return frame[y:y+h, x:x+w]
    

    def _get_features(self, cropped):
        """Return features."""
        gray = rgb2gray(cropped)
        gray = gray.reshape(*gray.shape, 1)
        ranks = [0.2, 0.4, 0.6, 0.8]
        layers = [(gray > rank) for rank in ranks]
        return np.concatenate(layers, axis=2).astype("float64")
