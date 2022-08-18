from ctypes import alignment

from manim_config import *

from ap_utils import *


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
        self.fns = []
        self.fns_segments_mobj = []

        self._function_set = None
        super().__init__(**kwargs)

    def construct(self):
        raise NotImplementedError

    def setup_scene(self):
        (
            self.true_dim,
            self.xtrue,
            self.ytrue,
            config,
        ) = simple_poly_regression_true_data()
        self.axes, self.axes_labels, _ = axes_and_data([], [], config, axes_labels=("x", "y"))
        VGroup(self.axes, self.axes_labels).scale(0.7).center()

        self.x_min = config["X_MIN"]
        self.x_max = config["X_MAX"]
        self.y_min = config["Y_MIN"]
        self.y_max = config["Y_MAX"]

        self.true_fn, self.true_fn_segments = degfungraph(
            self.axes, self.xtrue, self.ytrue, self.true_dim, COL_BLUE, config
        )
        self.true_flabel = BMathTex("f(x)", color=COL_BLUE)

        # move to the right of the axes
        self.true_flabel.move_to(
            self.true_fn.function(self.x_max)
            + RIGHT * 0.75
        )

        axes_and_fn_label = VGroup(self.axes, self.true_fn, self.true_flabel)

        self.play(Create(self.axes), Create(self.axes_labels))
        self.play(Create(self.true_fn_segments))
        self.play(Write(self.true_flabel))

    def fns_and_dots(self, seed):
        xs, ys, config = simple_poly_regression_get_data(seed=seed)
        fg, fg_segments = degfungraph(self.axes, xs, ys, self.degree, COL_PURPLE, config)
        dots = get_dots_for_axes(xs, ys, self.axes, config)

        return fg, fg_segments, dots

    def highlight_area_between_fn(self, f1, f2, dx=0.001, color=COL_RED, opacity=0.01):
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
        fg, fg_segments, dots = self.fns_and_dots(seed=1001001)
        self.fns.append(fg.underlying_function)
        self.fns_segments_mobj.append(fg_segments)

        flabel = BMathTex("f_{\hat{w}}(x)", color=COL_PURPLE)

        # move to the right of the axes
        flabel.move_to(
            fg.function(self.x_max) + RIGHT * 0.75
        )

        self.play(Create(dots))
        self.play(Create(fg_segments))
        self.play(Write(flabel))
        self.wait(0.5)
        self.play(fg_segments.animate.set_stroke(GRAY), FadeOut(dots), FadeOut(flabel))

    def slow_draw_functions(self):
        for i in range(1, self.nfirst):
            fg, fg_segments, dots = self.fns_and_dots(seed=1001001 + 147 * i)
            self.fns.append(fg.underlying_function)
            self.fns_segments_mobj.append(fg_segments)
            self.play(Create(dots), run_time=0.5)
            self.play(Create(fg_segments), run_time=0.5)
            self.play(fg_segments.animate.set_stroke(GRAY), FadeOut(dots))

    def fast_draw_functions(self):
        for i in range(self.nextra):
            fg, fg_segments, dots = self.fns_and_dots(seed=1101001 + 147 * i)
            fg_segments.set_stroke(GRAY)
            dots.set_color(GRAY)
            self.fns.append(fg.underlying_function)
            self.fns_segments_mobj.append(fg_segments)
            self.play(Create(dots), run_time=0.1)
            self.play(Create(fg_segments), run_time=0.1)
            self.play(FadeOut(dots), run_time=0.1)

    def remove_dots_and_fns(self):
        self.play(FadeOut(VGroup(*self.fns_segments_mobj)), run_time=0.2)

    def draw_mean_function(self):
        if not self._function_set:
            self._function_set = FunctionSet(self.fns)

        self.meanf, mean_segments = self.axes.plot_bounded(self._function_set.mean,
            x_range=(self.x_min, self.x_max),
            y_range=(self.y_min, self.y_max),
            color=COL_RED)

        meanf_label = BMathTex(r"\overline{f_{\hat{w}}}(x)", color=COL_RED)
        meanf_label.next_to(self.axes, UP + RIGHT)
        # meanf_label.move_to(min(self.meanf.function(self.x_max), self.y_max) + RIGHT * 0.75, aligned_edge=UP)


        self.play(Create(mean_segments))
        self.play(Write(meanf_label))

    def draw_variance_interval(self):
        if not self._function_set:
            self._function_set = FunctionSet(self.fns)

        # want to draw upper and lower function bounds for the variance
        self.upper_varf, upper_varf_segments = self.axes.plot_bounded(self._function_set.upper_confidence_bound,
            x_range=(self.x_min, self.x_max),
            y_range=(self.y_min, self.y_max),
            color=COL_GOLD)
        self.lower_varf, lower_varf_segments = self.axes.plot_bounded(self._function_set.lower_confidence_bound,
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
            Create(upper_varf_segments),
            Create(lower_varf_segments),
            Write(var_label),
        )
