from manim_config import *
from config import *


def true_error(x):
    mid = 5.0
    slope = 0.3 if x < mid else 0.1
    return slope * (x - mid) ** 2 + 2


def train_error(x):
    return 10 * np.exp(-x / 2.0 - 0.25) + 1


class BTrainTestScene(BScene):
    def set_axes_labels(self):
        axeslabels = []
        y_label = BText("Error", color=GRAY)
        y_label.scale(0.75)
        y_label.move_to(self.axes.c2p(self.x_min, self.y_max, 0) + UP * 0.25)

        x_label_low = BText(r"Low Model \\ Complexity", color=GRAY)
        x_label_low.scale(0.75)
        x_label_low.move_to(
            self.axes.c2p(self.x_min, self.y_min, 0) + DOWN * 0.5 + RIGHT
        )

        x_label_high = BText(r"High Model \\ Complexity", color=GRAY)
        x_label_high.scale(0.75)
        x_label_high.move_to(
            self.axes.c2p(self.x_max, self.y_min, 0) + DOWN * 0.5 + LEFT
        )

        self.axes_labels = [y_label, x_label_low, x_label_high]

    def setup_axes(self):
        self.axes, _ = axes_and_data([], [], GRAPH_CONFIG)

        self.x_min = GRAPH_CONFIG["X_MIN"]
        self.x_max = GRAPH_CONFIG["X_MAX"]
        self.y_min = GRAPH_CONFIG["Y_MIN"]
        self.y_max = GRAPH_CONFIG["Y_MAX"]

        self.set_axes_labels()

    def setup_scene(self):
        self.true_fn = FunctionOffGraph(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            function=true_error,
            color=COL_GOLD,
        )
        self.true_flabel = BText(r"True \\ Error", color=COL_GOLD)
        self.true_flabel.scale(0.75)

        # move to the right of the axes
        ypos = max(self.y_min, min(self.y_max, self.true_fn.function(self.x_max)))
        self.true_flabel.move_to(self.axes.c2p(self.x_max, ypos, 0) + RIGHT * 0.75)

        self.train_fn = FunctionOffGraph(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            function=train_error,
            color=COL_BLUE,
        )

        self.train_flabel = BText(r"Train \\ Error", color=COL_BLUE)
        self.train_flabel.scale(0.75)

        # move to the right of the axes
        ypos = max(self.y_min, min(self.y_max, self.train_fn.function(self.x_max)))
        self.train_flabel.move_to(self.axes.c2p(self.x_max, ypos, 0) + RIGHT * 0.75)

        self.axes_and_fn_label = VGroup(
            self.axes,
            self.true_fn,
            self.true_flabel,
            self.train_fn,
            self.train_flabel,
            *self.axes_labels
        )
        self.centershift = -self.axes_and_fn_label.get_center()
        self.axes_and_fn_label.move_to((0, 0, 0))

        self.play(ShowCreation(self.axes), Write(VGroup(*self.axes_labels)))
        self.play(ShowCreation(self.true_fn))
        self.play(Write(self.true_flabel))
        self.play(ShowCreation(self.train_fn))
        self.play(Write(self.train_flabel))
