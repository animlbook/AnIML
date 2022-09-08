from manim_config import *

from ap_utils import *

Y_RANGE = (0, 8.5)

class Animation(BScene):
    def construct(self):
        self.deg_col = [
            (1, COL_RED),
            (3, COL_GOLD),
            (5, COL_PURPLE),
            (7, COL_BLUE),
            (9, COL_GREEN),
        ]

        # Generate axes
        axes, axes_labels = make_bounded_axes(x_range=X_RANGE, y_range=Y_RANGE, axes_labels=("x", "y"))
        plot = VGroup(axes, axes_labels)
        plot.center().shift(LEFT).scale(0.8)

        # Generate data and dots
        Xs, ys = generate_train_data(x_range=X_RANGE, n=10, seed=100399)
        dots = get_dots_for_data(axes, Xs, ys, x_range=X_RANGE, y_range=Y_RANGE)

        # Make all of the individual degree plots
        fns = []
        labels = []

        for deg, color in self.deg_col:
            f_hat = train_and_plot_function(axes, Xs, ys, deg, x_range=X_RANGE, y_range=Y_RANGE, color=color)
            fns.append(f_hat)

            l = BTex(f"p = {deg}", color=color)
            if len(labels) > 0:
                l.next_to(labels[-1], DOWN / 2, aligned_edge=LEFT)

            labels.append(l)

        labels_grp = VGroup(*labels)
        labels_grp.next_to(plot, RIGHT)

        # Draw axes
        self.play(Create(plot))

        # Draw dots and then add them to foreground to make sure they are displayed above functions
        self.play(Create(dots))
        self.add_foreground_mobject(dots)

        # Animate in lines
        for f, l in zip(fns, labels):
            self.play(Write(l))
            self.play(Create(f.segments), run_time=1.5)
