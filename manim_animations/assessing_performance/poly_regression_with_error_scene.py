from operator import itemgetter

import numpy as np
from manim_config import *

from ap_utils import *

AXES_SCALE = 0.5
TEXT_SCALE = 0.5
STROKE_WIDTH = 2


class PolyRegressionWithErrorScene(BScene):
    def __init__(self, plot_title, error_title, error_fn, **kwargs):
        self.deg_col = [
            (1, COL_RED),
            (3, COL_GOLD),
            (5, COL_PURPLE),
            (7, COL_BLUE),
            (8, COL_GREEN),
            (10, GREY),
        ]

        self.xs, self.ys, self.config = simple_poly_regression_get_data(scale=0.6)
        self.axes, self.dots = axes_and_data(self.xs, self.ys, self.config, radius=0.05)


        # Position the function plot axes
        self.plot_grp = VGroup(self.axes, self.dots)
        self.plot_grp.scale(AXES_SCALE)

        # Make error plot axes
        x_range = (0, 0.4 / 0.6 * self.config["X_MAX"])
        self.error_plot = Axes(
            x_range=x_range,
            y_range=(0, self.config["Y_MAX"]),
            axis_config={"include_tip": False, "include_ticks": False, "color": GRAY},
        )
        self.error_plot.scale(AXES_SCALE)
        self.error_plot.next_to(self.plot_grp, RIGHT, buff=1.2)

        # Position the axes appropriately
        self.axes_grp = VGroup(self.plot_grp, self.error_plot)
        self.axes_grp.center()

        # Position function axes title
        self.plot_text = BText(plot_title)
        self.plot_text.next_to(self.plot_grp, 2 * UP).scale(TEXT_SCALE)

        # Make function plots
        self.fngraphs = list(
            map(
                lambda dc: degfungraph(self.axes, self.xs, self.ys, dc[0], dc[1], self.config, stroke_width=STROKE_WIDTH),
                self.deg_col,
            )
        )

        # Make error function plot
        self.error_fn = self.error_plot.plot(error_fn, x_range=(0.5, x_range[1]), color=BLACK, stroke_width=STROKE_WIDTH)

        # Position error plot title
        self.error_label = BText(error_title)
        self.error_label.next_to(self.error_plot, 2 * UP).scale(TEXT_SCALE)

        # Make points showing errors
        self.error_pts = VGroup()
        max_d = np.max(np.array(list(map(itemgetter(0), self.deg_col))))

        # hack so largest one isn't the too far to the right
        max_d += 2
        for d, c in self.deg_col:
            x = d / max_d * x_range[1]
            pt = (x, self.loss_fn(d), 0, 0)
            self.error_pts.add(Dot(self.error_plot.c2p(*pt), color=c)) #, radius=DEFAULT_DOT_RADIUS))

        self.error_grp = VGroup(self.error_plot, self.error_pts, self.error_fn)

        self.main_grp = VGroup(
            self.plot_grp,
            self.plot_text,
            *self.fngraphs,
            self.error_grp,
            self.error_label
        )

        super().__init__(**kwargs)

    def draw_residuals(self, fn):
        dots = VGroup()
        dashed = VGroup()
        for x, y in zip(self.xs, self.ys):
            yhat = fn.function(x)
            dots.add(Dot(self.axes.c2p(x, yhat, 0), color=GRAY))
            dashed.add(
                DashedLine(
                    self.axes.c2p(x, y, 0), self.axes.c2p(x, yhat, 0), color=GRAY
                )
            )

        self.play(ShowCreation(dots))
        self.play(ShowCreation(dashed))

    def loss_fn(self, x):
        raise NotImplementedError

    def construct(self):
        self.play(
            Create(self.plot_grp),
            Write(self.plot_text),
            Create(self.error_plot),
            Write(self.error_label),
        )

        # Add dots to foreground to avoid curves drawing over them
        self.add_foreground_mobject(self.dots)

        # Move to side by side graph


        # Play functions while adding the loss to the the train_loss function
        for fg, pt in zip(self.fngraphs, self.error_pts):
            self.play(Create(fg))
            self.play(Create(pt))

        self.play(Create(self.error_fn))
