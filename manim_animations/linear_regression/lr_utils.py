import numpy as np

from manim_config import *

# Config
X_MIN = 0
X_MAX = 4
Y_MIN = 0
Y_MAX = 4


# True function
def f(x):
    # Weird hacks since manim gives inputs of 0 to 4, but we want 0 to 1 for formula
    x = x / X_MAX
    return 21 * x ** 3 - 37 * x ** 2 + 20 * x


# Generated from random experiments to look "nice"
# ys = f(xs) + gaussian_noise
XS = np.array(
    [
        3.16087855,
        0.42209176,
        0.10747032,
        2.29757283,
        2.41755141,
        3.1312064,
        3.41274744,
        0.54062515,
        3.97612742,
        3.88264967,
        1.00726116,
        2.48506019,
        0.52647503,
        2.78625823,
        2.74012214,
        2.6109622,
        0.99758877,
        1.58532599,
        3.68280559,
        2.92428234,
    ]
)
YS = np.array(
    [
        3.01574732,
        1.15650066,
        0.29587366,
        3.63301317,
        2.44639242,
        4.01503537,
        2.72637157,
        1.83949441,
        4.05444247,
        3.45548138,
        3.48042945,
        2.81497412,
        2.11811096,
        3.47236632,
        3.05471791,
        3.37113595,
        2.48020868,
        3.33907243,
        3.05561223,
        2.30361556,
    ]
)


def make_axes():
    return Axes(
        x_range=(X_MIN, X_MAX),
        y_range=(Y_MIN, Y_MAX),
        height=5,
        width=5,
        axis_config={
            "stroke_color": GREY_D,
            "stroke_width": 2,
            "include_tip": False,
            "include_ticks": False,
        },
    )


class ModelScene(BScene):
    def custom_setup(self):
        # Create text
        self.data_text = BTex(r"(x_1, y_1), ..., (x_n, y_n)")

        self.function_text = BTex(
            r"\text{True function:}\ {{f(}}x{{)}}",
            tex_to_color_map={"f(": GREEN, ")": GREEN},
        )
        self.text_group = VGroup(self.data_text, self.function_text).arrange(DOWN)
        self.text_group.fix_in_frame()
        self.text_group.to_corner(UP + LEFT)

        # Create graph
        self.axes = make_axes()

        self.axes.next_to(self.text_group, BOTTOM + RIGHT, buff=0.25)
        self.function = self.axes.get_graph(f, color=GREEN)
        self.graph = Group(self.axes, self.function)

        # Draw points
        self.dots = Group()
        for x, y in zip(XS, YS):
            point = self.axes.c2p(x, y)
            dot = Dot(point, color=COL_BLACK)
            self.dots.add(dot)


class LinearScene(ModelScene):
    def custom_setup(self, line_color, **kwargs):
        super().custom_setup(**kwargs)

        # Create text for predictor
        self.predictor_text = BTex(
            r"\text{Predictor:}\ \hat{f}(x)",
            tex_to_color_map={"\\hat{f}(": BLUE, ")": BLUE},
        )
        self.predictor_text.next_to(self.text_group, DOWN)

        # Create line
        self.w_0, self.w_1 = 2, 0.4
        self.line = self.linear_function(self.w_0, self.w_1, line_color)

    def linear_function(self, w_0, w_1, line_color):
        """
        Returns a manim Line for the given linear function (translated to self.axes)
        """

        start = self.axes.coords_to_point(0, w_0, 0)
        end = self.axes.coords_to_point(X_MAX, w_0 + w_1 * X_MAX, 0)
        line = Line(start=start, end=end, color=line_color)
        return line


class GradientDescentScene(BScene):
    def custom_setup(self, f, start_x, end_x):
        # Must create the functions/axes before everything else so they all move together

        # Create axes
        axes = make_axes()

        # Draw graph of original function
        function = axes.get_graph(f, color=GREEN)

        # Create axis labels
        x_label = axes.get_x_axis_label(r"w_1")
        y_label = axes.get_y_axis_label(r"RSS(w_1)")
        self.add(axes, x_label, y_label)

        # Draw function
        self.play(ShowCreation(function))

        # Draw start point
        dot = Dot(color=BLUE)
        dot.move_to(axes.i2gp(start_x, function))
        self.play(FadeIn(dot))

        # Move the point down the function
        x_tracker = ValueTracker(start_x)
        f_always(dot.move_to, lambda: axes.i2gp(x_tracker.get_value(), function))

        self.play(x_tracker.animate.set_value(end_x), run_time=2.5)

        self.wait(3)
