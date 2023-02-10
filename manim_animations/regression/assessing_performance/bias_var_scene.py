from ctypes import alignment

from ap_utils import *

from manim_config import *


class FunctionSet:
    """
    Represents a set of functions and com
    """
    def __init__(self, functions):
        self.functions = functions

        # For efficiency, we should memo-ize the results so we don't have to recompute.
        self._means = {}
        self._upper_confidence = {}
        self._lower_confidence = {}

    def mean(self, x):
        if x in self._means:
            return self._means[x]
        else:
            val = 0.0
            for f in self.functions:
                val += f(x)

            result = val / len(self.functions)
            self._means[x] = result
            return result

    def _confidence_bound(self, x, upper: bool):
        if upper and x in self._upper_confidence:
            return self._upper_confidence[x]
        elif not upper and x in self._lower_confidence:
            return self._lower_confidence[x]
        else:
            value = 0.0
            mean = self.mean(x)
            for f in self.functions:
                value += pow(f(x) - mean, 2)

            up_result = mean + (value / len(self.functions))
            down_result = mean - (value / len(self.functions))

            self._upper_confidence[x] = up_result
            self._lower_confidence[x] = down_result

            if upper:
                return up_result
            else:
                return down_result

    def upper_confidence_bound(self, x):
        return self._confidence_bound(x, True)

    def lower_confidence_bound(self, x):
        return self._confidence_bound(x, False)


class BBiasVarianceScene(BScene):

    def __init__(self, degree, nfirst, nextra, **kwargs):
        self.nfirst = nfirst
        self.nextra = nextra
        self.degree = degree
        self.functions = []

        self._function_set = None
        super().__init__(**kwargs)

    def construct(self):
        raise NotImplementedError()

    def setup_scene(self):
        self.x_min = X_RANGE[0]
        self.x_max = X_RANGE[1]
        self.y_min = Y_RANGE[0]
        self.y_max = Y_RANGE[1]

        # Generate axes
        self.axes, self.axes_labels = make_bounded_axes(x_range=X_RANGE, y_range=Y_RANGE,
            axes_labels=("x", "y"))
        VGroup(self.axes, self.axes_labels).scale(0.7).center()

        # Make true function
        self.true_fn = self.axes.plot_bounded(
            lambda x: true_function(x)[0],
            x_range=X_RANGE, y_range=Y_RANGE,
            color=COL_BLUE)

        self.true_flabel = BMathTex("f(x)", color=COL_BLUE)

        # move to the right of the axes
        self.true_flabel.move_to(
            self.axes.function_label_pos(self.true_fn.underlying_function, self.x_max,
                y_range=(self.y_min, self.y_max))
            + RIGHT * 0.75
        )

        self.play(Create(self.axes), Create(self.axes_labels))
        self.play(Create(self.true_fn.segments))
        self.play(Write(self.true_flabel))

    def fns_and_dots(self, seed):
        Xs, ys = generate_train_data(x_range=X_RANGE, clip_y_range=Y_RANGE,
            seed=seed)

        function = train_and_plot_function(self.axes, Xs, ys,
            x_range=X_RANGE, y_range=Y_RANGE,
            deg=self.degree, color=COL_PURPLE)

        dots = get_dots_for_data(self.axes, Xs, ys)
        return function, dots

    def highlight_area_between_fn(self, f1, f2, dx=0.01, color=COL_RED, opacity=0.01):
        xs = np.arange(self.x_min + 0.015, self.x_max, dx) + (dx / 2.0)
        rects = VGroup()
        for x in xs:
            y1 = f1.underlying_function(x)
            y2 = f2.underlying_function(x)

            x_left = x - dx / 2
            x_right = x + dx / 2
            upper = min(max(y1, y2), self.y_max)
            lower = max(min(y1, y2), self.y_min)

            up_left = np.array([x_left, upper, 0])
            up_right = np.array([x_right, upper, 0])
            down_left = np.array([x_left, lower, 0])
            down_right = np.array([x_right, lower, 0])

            rects.add(
                Polygon(
                    self.axes.c2p(*up_left),
                    self.axes.c2p(*up_right),
                    self.axes.c2p(*down_right),
                    self.axes.c2p(*down_left),
                    color=color,
                    stroke_opacity=opacity,
                )
            )
        self.play(Create(rects))
        self.wait(0.5)

    def explain_first_function(self):
        function, dots = self.fns_and_dots(seed=1001001)
        self.functions.append(function)

        flabel = BMathTex("f_{\hat{w}}(x)", color=COL_PURPLE)

        # move to the right of the axes
        flabel.move_to(
            self.axes.function_label_pos(function.underlying_function, self.x_max,
                y_range=(self.y_min, self.y_max))
            + RIGHT * 0.75
        )

        self.play(Create(dots))
        self.play(Create(function.segments))
        self.play(Write(flabel))
        self.wait(0.5)
        self.play(function.segments.animate.set_stroke(GRAY), FadeOut(dots), FadeOut(flabel))

    def slow_draw_functions(self):
        for i in range(1, self.nfirst):
            function, dots = self.fns_and_dots(seed=1001001 + 147 * i)
            self.functions.append(function)
            self.play(Create(dots), run_time=0.5)
            self.play(Create(function.segments), run_time=0.5)
            self.play(function.segments.animate.set_stroke(GRAY), FadeOut(dots))

    def fast_draw_functions(self):
        for i in range(self.nextra):
            function, dots = self.fns_and_dots(seed=1101001 + 147 * i)
            function.segments.set_stroke(GRAY, opacity=0.5)

            dots.set_color(GRAY)
            self.functions.append(function)
            self.play(Create(dots), run_time=0.1)
            self.play(Create(function.segments), run_time=0.1)
            self.play(FadeOut(dots), run_time=0.1)

    def remove_dots_and_fns(self):
        mobjects = [f.segments for f in self.functions]
        self.play(FadeOut(VGroup(*mobjects)), run_time=0.75)

    def draw_mean_function(self, label_loc=None):
        if not self._function_set:
            functions = [f.underlying_function for f in self.functions]
            self._function_set = FunctionSet(functions)

        self.meanf = self.axes.plot_bounded(self._function_set.mean,
            x_range=(self.x_min, self.x_max),
            y_range=(self.y_min, self.y_max),
            color=COL_RED)

        meanf_label = BMathTex(r"\overline{f_{\hat{w}}}(x)", color=COL_RED)
        if label_loc is None:
            pt = self.axes.function_label_pos(self._function_set.mean, self.x_max,
                y_range=(self.y_min, self.y_max))
            meanf_label.move_to(pt + RIGHT * 0.75, aligned_edge=UP)
        else:
            meanf_label.next_to(self.axes, label_loc)

        self.play(Create(self.meanf.segments))
        self.play(Write(meanf_label))

    def draw_variance_interval(self):
        if not self._function_set:
            functions = [f.underlying_function for f in self.functions]
            self._function_set = FunctionSet(functions)

        # want to draw upper and lower function bounds for the variance
        self.upper_varf = self.axes.plot_bounded(self._function_set.upper_confidence_bound,
            x_range=(self.x_min, self.x_max),
            y_range=(self.y_min, self.y_max),
            color=COL_GOLD)
        self.lower_varf = self.axes.plot_bounded(self._function_set.lower_confidence_bound,
            x_range=(self.x_min, self.x_max),
            y_range=(self.y_min, self.y_max),
            color=COL_GOLD)

        var_label = BMathTex(
            r"\overline{\left(\overline{f_{\hat{w}}}(x) - f_{\hat{w}}(x)\right)^2}",
            color=COL_GOLD,
        )
        var_label.next_to(self.axes, DOWN + RIGHT, aligned_edge=RIGHT)
        # var_label.move_to(self.axes.c2p(self.x_max, self.y_min - 0.475, 0) + RIGHT * 0.75, aligned_edge=LEFT)

        self.play(
            Create(self.upper_varf.segments),
            Create(self.lower_varf.segments),
            Write(var_label),
        )
