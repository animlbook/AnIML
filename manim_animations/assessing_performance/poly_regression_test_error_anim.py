from config import *
from poly_regression_with_error_scene import PolyRegressionWithErrorScene


def error_fn(x):
    y = 0.5 * (x - 2.25) ** 2 + 1
    return y


class Animation(PolyRegressionWithErrorScene):
    def __init__(self, **kwargs):
        super().__init__("Training Data", "True Error", error_fn, **kwargs)

    def loss_fn(self, x):
        Ntest = 1000
        testx = np.linspace(0, (0.4 / 0.6) * self.config["X_MAX"], Ntest)
        deg, tx, ty, _ = simple_poly_regression_true_data()
        yapprox = fhat_vector(testx, tx, ty, deg)

        fh = fhat_vector(testx, self.xs, self.ys, int(x))
        int_approx = np.square(np.linalg.norm(fh - yapprox)) / Ntest
        return int_approx / 5
