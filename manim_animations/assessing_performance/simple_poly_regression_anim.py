from manim_config import *

from ap_utils import *


class Animation(BScene):
    def construct(self):
        self.deg_col = [
            (1, COL_RED),
            (3, COL_GOLD),
            (5, COL_PURPLE),
            (7, COL_BLUE),
            (8, COL_GREEN),
        ]

        xs, ys, config = simple_poly_regression_get_data(seed=100399, N=10)
        axes, dots = axes_and_data(xs, ys, config)

        # Position the plot
        plot = VGroup(axes, dots)
        plot.center().shift(LEFT)

        # Make all of the individual degree plots
        fns = []
        labels = []

        for deg, col in self.deg_col:
            fn, segments = degfungraph(axes, xs, ys, deg, col, config)
            fns.append(segments)

            l = BTex(f"p = {deg}", color=col)
            if len(labels) > 0:
                l.next_to(labels[-1], DOWN / 2, aligned_edge=LEFT)

            labels.append(l)

        fns_group = VGroup(*fns)
        labels_grp = VGroup(*labels)
        labels_grp.next_to(plot, RIGHT)

        # Add dots to foreground to make sure they are displayed above functions
        self.add_foreground_mobject(dots)

        # Animate in
        self.play(Create(plot))
        for f, l in zip(fns, labels):
            self.play(Write(l))
            self.play(Create(f), run_time=1.5)
