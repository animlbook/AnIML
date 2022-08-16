from config import *
from poly_regression_with_error_scene import PolyRegressionWithErrorScene


def error_fn(x):
    y = 5 * np.exp(-x) + 0.5
    return y


class Animation(PolyRegressionWithErrorScene):
    def __init__(self, **kwargs):
        super().__init__("Training Data", "Training Error", error_fn, **kwargs)

    def loss_fn(self, x):
        fh = fhat_vector(self.xs, self.xs, self.ys, int(x))
        # kludge: shifting for a better look on graph
        return np.square(np.linalg.norm(self.ys - fh)) / 5
