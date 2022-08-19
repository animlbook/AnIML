from operator import itemgetter

import numpy as np
from manim_config import *

from ap_utils import *

AXES_SCALE = 0.5
TEXT_SCALE = 0.5
STROKE_WIDTH = 2

DATA_X_RANGE = (0, 12)
DATA_Y_RANGE = (0, 8)

ERROR_X_RANGE = (0, 10, .1)  # Needs to match degree polynomials + 1
ERROR_Y_RANGE = (0, 8)


class PolyRegressionWithErrorScene(BScene):
    def __init__(self, data_plot_title, error_plot_title, error_fn, error_y_range=ERROR_Y_RANGE, **kwargs):
        self.deg_col = [
            (1, COL_RED),
            (3, COL_GOLD),
            (5, COL_PURPLE),
            (7, COL_BLUE),
            (9, COL_GREEN),
        ]

        # should be sorted by degree
        assert all(self.deg_col[i][0] < self.deg_col[i + 1][0]
                   for i in range(len(self.deg_col) - 1))

        # Make data
        self.Xs, self.ys = generate_train_data(x_range=DATA_X_RANGE, n=10, seed=100399)

        # Make axes for data and errors
        self.data_axes, self.data_axes_labels = make_bounded_axes(x_range=DATA_X_RANGE, y_range=DATA_Y_RANGE,
                axes_labels=("x", "y"))
        self.error_axes, self.error_axes_labels = make_bounded_axes(x_range=ERROR_X_RANGE, y_range=error_y_range,
                axes_labels=("p", "Error"), axes_labels_rotations=(0, 90), y_label_buff=0.25)

        # Make groups for plots and position them
        self.data_plot_grp = VGroup(self.data_axes, self.data_axes_labels)
        self.data_plot_grp.scale(AXES_SCALE)
        self.error_plot_grp = VGroup(self.error_axes, self.error_axes_labels)

        self.error_plot_grp.scale(AXES_SCALE)
        self.error_plot_grp.next_to(self.data_plot_grp, RIGHT, buff=1.2)

        # Position both axes to be centered
        self.both_plots = VGroup(self.data_plot_grp, self.error_plot_grp)
        self.both_plots.center()

        # Generate data
        self.data_dots = get_dots_for_data(self.data_axes, self.Xs, self.ys, x_range=DATA_X_RANGE)
        self.data_plot_grp.add(self.data_dots)

        # Make plot titles
        self.data_plot_title = BText(data_plot_title)
        self.data_plot_title.next_to(self.data_plot_grp, 2 * UP).scale(TEXT_SCALE)

        self.error_plot_title = BText(error_plot_title)
        self.error_plot_title.next_to(self.error_plot_grp, 2 * UP).scale(TEXT_SCALE)

        # Make function plots
        self.model_graphs = []
        for deg, color in self.deg_col:
            f_hat = train_and_plot_function(self.data_axes,
                    self.Xs, self.ys, deg,
                    x_range=DATA_X_RANGE, y_range=DATA_Y_RANGE,
                    color=color,
                    stroke_width=STROKE_WIDTH)
            self.model_graphs.append(f_hat)

        # Make error function plot
        self.error_fn = self.error_axes.plot_bounded(error_fn,
                x_range=ERROR_X_RANGE, y_range=ERROR_Y_RANGE,
                color=BLACK,
                stroke_width=STROKE_WIDTH)

        # Make error dots
        self.error_dots = VGroup()
        for d, c in self.deg_col:
            pt = (d, self.loss_fn(d), 0, 0)
            self.error_dots.add(Dot(self.error_axes.c2p(*pt), color=c))


        super().__init__(**kwargs)


    def loss_fn(self, x):
        raise NotImplementedError

    def construct(self):
        self.play(
            Create(self.data_plot_grp),
            Write(self.data_plot_title),
            Create(self.error_plot_grp),
            Write(self.error_plot_title),
        )

        # Add dots to foreground to avoid curves drawing over them
        self.add_foreground_mobject(self.data_dots)

        # Play functions while adding the loss to the the train_loss function
        for f, pt in zip(self.model_graphs, self.error_dots):
            self.play(Create(f.segments))
            self.play(Create(pt))

        self.play(Create(self.error_fn.segments))
