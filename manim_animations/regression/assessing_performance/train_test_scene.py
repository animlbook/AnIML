import numpy as np
from ap_utils import *

from manim_config import *

LABEL_BUFF = 0.1

X_RANGE = (1, 12)
Y_RANGE = (0, 6)

np.random.seed(10)

class BTrainTestScene(BScene):
    @staticmethod
    def true_error(x):
        mid = 5.0
        slope = 0.3 if x < mid else 0.1
        return slope * (x - mid) ** 2 + 2

    @staticmethod
    def train_error(x):
        return 10 * np.exp(-x / 2.0 - 0.25) + 1

    def test_error(self, x):
        # Keeps track of a random walk, centered at 0 plus the true error
        #self._test_noise += np.random.normal(loc=0, scale=0.01)

        # Try to add a noisy sine wave, but it gets weird at the edges so reduce the coefficients near
        # the boundaries
        y_mid = (X_RANGE[0] + X_RANGE[1]) / 2
        sine_noise_scale = 0.5 * np.exp(-((x - y_mid) / 10) ** 2)
        sine_noise = sine_noise_scale * np.random.normal(0, scale=sine_noise_scale) * np.sin(2 * x)
        return BTrainTestScene.true_error(x) + sine_noise

    def set_axes_labels(self):
        y_label = BText("Error", color=GRAY)
        y_label.scale(0.6)
        y_label.next_to(self.axes, LABEL_BUFF * (UP + LEFT), aligned_edge=UP, buff=1)

        x_label_low = BTex(r"Low Model \\ Complexity", color=GRAY)
        x_label_low.scale(0.6)
        x_label_low.next_to(self.axes, LABEL_BUFF * (DOWN + LEFT), aligned_edge=LEFT, buff=1)

        x_label_high = BTex(r"High Model \\ Complexity", color=GRAY)
        x_label_high.scale(0.6)
        x_label_high.next_to(self.axes, LABEL_BUFF * (DOWN + RIGHT), aligned_edge=RIGHT, buff=1)

        self.axes_labels = [y_label, x_label_low, x_label_high]

    def setup_axes(self):
        self.axes = make_bounded_axes(x_range=X_RANGE, y_range=Y_RANGE)

        self.axes.scale(0.7)
        self.axes.center()

    def setup_scene(self):
        self.setup_axes()
        self.set_axes_labels()

        self.true_fn = self.axes.plot_bounded(self.true_error,
            x_range=X_RANGE,
            y_range=Y_RANGE,
            color=COL_GOLD)

        self.true_flabel = BTex(r"True \\ Error", color=COL_GOLD)
        self.true_flabel.scale(0.75)

        # move to the right of the axes
        pos = self.axes.function_label_pos(self.true_error, X_RANGE[1],
            y_range=Y_RANGE)
        self.true_flabel.move_to(pos + RIGHT * 0.75)

        # Test error
        self._test_noise  = 0  # Need to set up how much the test differs from true here
        self.test_fn = self.axes.plot_bounded(lambda x: self.test_error(x),
            x_range=X_RANGE,
            y_range=Y_RANGE,
            color=COL_RED)

        self.test_flabel = BTex(r"Test \\ Error", color=COL_RED)
        self.test_flabel.scale(0.75)

        # TODO test_errors
        self.test_flabel.move_to(pos + 0.75 * RIGHT + DOWN)

        # Set up train error curve
        self.train_fn = self.axes.plot_bounded(self.train_error,
            x_range=X_RANGE,
            y_range=Y_RANGE,
            color=COL_BLUE)

        self.train_flabel = BTex(r"Train \\ Error", color=COL_BLUE)
        self.train_flabel.scale(0.75)

        # move to the right of the axes
        pos = self.axes.function_label_pos(self.train_error, X_RANGE[1],
            y_range=Y_RANGE)
        self.train_flabel.move_to(pos + RIGHT * 0.75)

        self.axes_and_fn_label = VGroup(
            self.axes,
            self.true_fn,
            self.true_flabel,
            self.test_fn,
            self.test_flabel,
            self.train_fn,
            self.train_flabel,
            *self.axes_labels
        )
        #self.centershift = -self.axes_and_fn_label.get_center()
        #self.axes_and_fn_label.move_to((0, 0, 0))

        self.play(Create(self.axes), Write(VGroup(*self.axes_labels)))
        self.play(Create(self.true_fn.segments))
        self.play(Write(self.true_flabel))
        self.play(Create(self.test_fn.segments))
        self.play(Write(self.test_flabel))
        self.play(Create(self.train_fn.segments))
        self.play(Write(self.train_flabel))
