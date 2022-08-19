from ap_utils import *
from poly_regression_with_error_scene import PolyRegressionWithErrorScene

TRAIN_ERROR_Y_RANGE = (0, 3)

def error_fn(x):
    # All magic numbers to make the curve look nice
    y = 1.5 * np.exp(-0.5 * x) + 0.1
    return y


class Animation(PolyRegressionWithErrorScene):
    def __init__(self, **kwargs):
        super().__init__("Training Data", "Training Error", error_fn,
                error_y_range=TRAIN_ERROR_Y_RANGE, **kwargs)

    def loss_fn(self, deg):
        w_hat = train(self.Xs, self.ys, deg)
        y_hat = predict(self.Xs, w_hat)
        return np.square(np.linalg.norm(self.ys - y_hat)) / len(self.Xs)
        # Kludge: divide by 5?

