# Manim config has a re-definition of Union
import typing
from typing import Sequence, Union

import numpy as np
from manim_config import *

GRAPH_CONFIG = {
    "X_MIN": 0,
    "X_MAX": 12,
    "Y_MIN": 0,
    "Y_MAX": 6,
}

X_RANGE = np.array([0, 12])
Y_RANGE = np.array([0, 6])

SegmentedParametricFunction = ParametricFunction

class BoundedAxes(Axes):
    def plot_bounded(self, function, x_range=None, y_range=None, use_vectorized=None, **kwargs) -> SegmentedParametricFunction:
        """
        Same as plot, but hides regions with values outside the y-range
        """
        # TODO Ignores use_vectorized at the moment

        min_y, max_y = y_range[:2]
        min_x, max_x = x_range[:2]

        # Figure out evaluation deltas
        if "dt" in kwargs:
            dt = kwargs["dt"]
        elif len(x_range) == 3:
            dt = x_range[2]
        else:
            dt = 0.01

        # Go through and find the distinct ranges of xs where the output is within the y range
        ranges = []
        start_curr_range = None

        curr_x = min_x
        while curr_x < max_x:
            curr_y = function(curr_x)

            if curr_y < min_y or curr_y > max_y:
                if start_curr_range is not None:
                    ranges.append((start_curr_range, curr_x - dt))  # Want to stop one before
                    start_curr_range = None
            else:
                if start_curr_range is None:
                    start_curr_range = curr_x

            curr_x += dt

        if start_curr_range is not None:
            ranges.append((start_curr_range, curr_x))

        # Go through and make plots for all of the discrete sections
        segments = VGroup(*[
            self.plot(function, x_range=sub_x_range, use_vectorized=use_vectorized, **kwargs)
            for sub_x_range in ranges
        ])

        # Want to also return the "full" function for usability
        full_function = self.plot(function, x_range=x_range, use_vectorized=use_vectorized, **kwargs)

        full_function.segments = segments

        return full_function

    def function_label_pos(self, f, x, y_range):
        y_min, y_max = y_range
        y = f(x)
        y = min(y, y_max)
        y = max(y, y_min)

        return self.c2p(x, y)



def polynomial_expand(x: typing.Union[float, Sequence[float]] , deg: int) -> np.ndarray:
    if np.isscalar(x):
        features = np.zeros((1, deg + 1))
        for i in range(deg + 1):
            features[0, i] = pow(x, i)
        return features
    else:  # Is list of xs
        n = x.shape[0]
        new_X = np.empty((n, deg + 1))
        for i in range(n):
            new_X[i] = polynomial_expand(x[i], deg)[0, :]  # Recurse one level to expand single x
        return new_X


def predict(x, coeffs):
    x = polynomial_expand(x, len(coeffs) - 1)
    result = x @ coeffs
    return result


def true_function(x):
    dim = 6  # Includes intercept
    true_coeffs = np.array([
        -9.77643998e+00,
        1.77393973e+01,
        -7.34515366e+00,
        1.27881487e+00,
        -9.94748672e-02,
        2.85635347e-03
    ]).reshape((dim, 1))
    return predict(x, true_coeffs)


def generate_train_data(x_range=None, Xs=None, n=25, noise=0.6, seed=100394, clip_y_range=None):
    """
    Generates n data points on the range x_range using given seed.

    If clip_y_range is specified, will clip any examples outside of the range
    """
    assert (x_range is None and Xs is not None) or (x_range is not None and Xs is None)

    np.random.seed(seed)

    if x_range is not None:  # Xs is None
        x_min, x_max = x_range
        Xs = np.random.uniform(x_min, x_max, n).reshape((n, 1))
    ys = true_function(Xs) + np.random.normal(0, noise, size=(len(Xs), 1))

    if clip_y_range is not None:
        y_min, y_max = clip_y_range
        valid_mask = (ys >= y_min) & (ys <= y_max)
        Xs = Xs[valid_mask][:, None]
        ys = ys[valid_mask][:, None]

    return Xs, ys

def train(X_train, y_train, deg):
    assert y_train.shape[1] == 1

    X_train = polynomial_expand(X_train, deg)
    w_hat = np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), np.matmul(X_train.T, y_train))

    return w_hat


# Manim code
def make_bounded_axes(x_range, y_range, axes_labels=None, axes_labels_rotations=None, y_label_buff=0):
    axes = BoundedAxes(
        x_range=x_range,
        y_range=y_range,
        axis_config={"include_tip": False, "include_ticks": False, "color": GREY_C},
    )

    if axes_labels:
        x_label_text, y_label_text = axes_labels

        if axes_labels_rotations:
            x_label_rotation, y_label_rotation = axes_labels_rotations
        else:
            x_label_rotation, y_label_rotation = 0, 0

        axes_labels_grp = VGroup()

        if x_label_text:
            x_label = axes.get_x_axis_label(x_label_text, edge=DOWN)\
                    .shift(0.75 * DOWN)\
                    .rotate(x_label_rotation * DEGREES)\
                    .set_color(GRAY_C)
            axes_labels_grp.add(x_label)

        if y_label_text:
            y_label = axes.get_y_axis_label(y_label_text, edge=RIGHT, direction=LEFT, buff=0.4)\
                    .set_color(GRAY_C)\
                    .rotate(y_label_rotation * DEGREES)\
                    .shift(y_label_buff * LEFT)
            axes_labels_grp.add(y_label)

        return axes, axes_labels_grp
    else:
        return axes


def get_dots_for_data(axes, Xs, ys, x_range=None, y_range=None, radius=DEFAULT_DOT_RADIUS):
    dots = VGroup()
    for x, y in zip(Xs[:, 0], ys[:, 0]):
        valid_x = x_range is None or (x >= x_range[0] and x <= x_range[1])
        valid_y = y_range is None or (y >= y_range[0] and y <= y_range[1])
        if valid_x and valid_y:
            point = axes.coords_to_point(x, y, 0)
            dot = Dot(point, color=COL_BLACK, radius=radius)
            dots.add(dot)

    return dots


def train_and_plot_function(axes, X_train, y_train, deg, x_range, y_range, color, **kwargs):
    assert y_train.shape[1] == 1
    assert y_train.shape[0] == len(X_train)

    w_hat = train(X_train, y_train, deg)

    def f(x):
        return predict(x, w_hat)[0, 0]

    return axes.plot_bounded(f, x_range=x_range, y_range=y_range,
        color=color, **kwargs)

