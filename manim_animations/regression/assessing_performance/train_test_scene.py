import numpy as np
from ap_utils import *
from colour import Color

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

    @staticmethod
    def test_error(x):
        # Try to add a noisy sine wave, but it gets weird at the edges so reduce the coefficients near
        # the boundaries
        y_mid = (X_RANGE[0] + X_RANGE[1]) / 2
        sine_noise_scale = 0.5 * np.exp(-((x - y_mid) / 10) ** 2)
        sine_noise = sine_noise_scale * np.random.normal(0, scale=sine_noise_scale) * np.sin(2 * x)
        return BTrainTestScene.true_error(x) + sine_noise

    def set_axes_labels(self):
        y_label = BTex("Error", color=GRAY)
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


    def _plot_curve(self, fn: Callable[[float], float], label_text: str, label_decorator: VMobject, color: Color,
            stroke_opacity=1):
        # Plot function
        fn_plot = self.axes.plot_bounded(fn,
            x_range=X_RANGE,
            y_range=Y_RANGE,
            color=color,
            stroke_opacity=stroke_opacity)

        # Make label
        fn_label = BTex(label_text, color=color)
        fn_label.scale(0.75)

        # Position label decorator
        label_decorator.set_color(color)
        label_decorator.next_to(fn_label, RIGHT, buff=0.25)

        # Position label and its decorator
        label_group = VGroup(fn_label, label_decorator)

        label_pos = self.axes.function_label_pos(fn, X_RANGE[1],
            y_range=Y_RANGE)
        label_group.move_to(label_pos + RIGHT)

        return fn_plot, label_group


    def setup_scene(self):
        self.setup_axes()
        self.set_axes_labels()

        # Training curve
        self.true_fn, self.true_fn_label = self._plot_curve(self.true_error,
                label_text=r"True \\ Error",
                color=COL_GOLD,
                label_decorator=Line([0, 0, 0], [0.5, 0, 0]))


        # Test Curve (has complex label decorator)
        p1 = np.array([-1, 0.5, 0])
        p2 = np.array([1, -0.5, 0])
        test_label_decorator = CubicBezier(p1, p1 + 1 * RIGHT, p2 - 1 * RIGHT, p2, color=COL_RED)
        test_label_decorator.scale(0.33)

        self.test_fn, self.test_fn_label = self._plot_curve(self.test_error,
                label_text=r"Test \\ Error",
                color=COL_RED,
                label_decorator=test_label_decorator,
                stroke_opacity=0.8)

        # Need to move the test label down a bit since it overlaps with the true error label
        self.test_fn_label.next_to(self.true_fn_label, DOWN)

        # Set up train error curve
        self.train_fn, self.train_fn_label = self._plot_curve(self.train_error,
                label_text=r"Train \\ Error",
                color=COL_BLUE,
                label_decorator=Line([0, 0, 0], [0.5, 0, 0]),)

        self.axes_and_fn_label = VGroup(
            self.axes,
            self.true_fn,
            self.true_fn_label,
            self.test_fn,
            self.test_fn_label,
            self.train_fn,
            self.train_fn_label,
            *self.axes_labels
        )

        self.play(Create(self.axes), Write(VGroup(*self.axes_labels)))
        self.play(Create(self.true_fn.segments))
        self.play(Write(self.true_fn_label))
        self.play(Create(self.test_fn.segments))
        self.play(Write(self.test_fn_label))
        self.play(Create(self.train_fn.segments))
        self.play(Write(self.train_fn_label))
