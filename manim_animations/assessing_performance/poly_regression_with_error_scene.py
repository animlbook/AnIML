from manim_config import *
from config import *
import numpy as np
from operator import itemgetter


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
        self.axes, dots = axes_and_data(self.xs, self.ys, self.config, radius=0.05)

        self.plot_grp = VGroup(self.axes, dots)

        self.plot_text = BText(plot_title)
        self.plot_text.next_to(self.plot_grp, 2 * UP)

        self.fngraphs = list(
            map(
                lambda dc: degfungraph(self.xs, self.ys, dc[0], dc[1], self.config),
                self.deg_col,
            )
        )

        # construct error plot
        self.error_plot = Axes(
            x_min=0,
            y_min=0,
            x_max=(0.4 / 0.6) * self.config["X_MAX"],
            y_max=self.config["Y_MAX"],
            axis_config={"include_tip": False, "include_ticks": False, "color": GRAY},
        )

        self.error_fn = FunctionOffGraph(
            x_min=0,
            y_min=0,
            x_max=(0.4 / 0.6) * self.config["X_MAX"],
            y_max=self.config["Y_MAX"],
            function=error_fn,
            color=COL_BLACK,
        )
        self.error_plot_and_fn = VGroup(self.error_plot, self.error_fn)
        self.error_plot_and_fn.next_to(self.plot_grp, 4 * RIGHT)

        self.error_label = BText(error_title)
        self.error_label.move_to(
            (self.error_plot.get_center()[0], self.plot_text.get_center()[1], 0)
        )

        self.error_pts = VGroup()
        max_d = np.max(np.array(list(map(itemgetter(0), self.deg_col))))
        # hack so largest one isn't the too far to the right
        max_d += 2
        for d, c in self.deg_col:
            x = d / max_d * (0.4 / 0.6) * self.config["X_MAX"]
            pt = (x, self.loss_fn(d), 0)
            self.error_pts.add(Dot(self.error_plot.c2p(*pt), color=c))

        self.error_grp = VGroup(self.error_plot, self.error_pts, self.error_fn)

        self.main_grp = VGroup(
            self.plot_grp,
            self.plot_text,
            *self.fngraphs,
            self.error_grp,
            self.error_label
        )

        self.main_grp.move_to((0, 0, 0))

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
        # Move to side by side graph
        self.play(
            ShowCreation(self.plot_grp),
            Write(self.plot_text),
            ShowCreation(self.error_plot),
            Write(self.error_label),
        )

        # Play functions while adding the loss to the the train_loss function
        for fg, pt in zip(self.fngraphs, self.error_pts):
            self.play(ShowCreation(fg))
            self.play(ShowCreation(pt))

        self.play(ShowCreation(self.error_fn))
