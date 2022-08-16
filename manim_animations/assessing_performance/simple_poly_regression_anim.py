from manim_config import *
from config import *


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

        plot = VGroup(axes, dots)
        fns = []
        labels = []

        for deg, col in self.deg_col:
            fns.append(degfungraph(xs, ys, deg, col, config))

            l = BTex(f"p = {deg}", color=col)
            if len(labels) > 0:
                l.next_to(labels[-1], DOWN / 2, aligned_edge=LEFT)

            labels.append(l)

        labels_grp = VGroup(*labels)

        graph_and_fns_grp = VGroup(plot, *fns)
        labels_grp.next_to(graph_and_fns_grp, RIGHT)

        graph_fns_labels_grp = VGroup(graph_and_fns_grp, labels_grp)
        # model_text.next_to(graph_fns_labels_grp, UP)

        grp = VGroup(graph_fns_labels_grp)  # , model_text)
        grp.move_to((0, 0, 0))

        self.play(ShowCreation(plot))  # , ShowCreation(model_text))
        for f, l in zip(fns, labels):
            self.play(Write(l))
            self.play(ShowCreation(f), run_time=1.5)
