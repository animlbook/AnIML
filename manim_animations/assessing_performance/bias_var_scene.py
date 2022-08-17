from ctypes import alignment

from manim_config import *

from ap_utils import *


class FunctionSet:
    """
    Represents a set of functions and com
    """
    def __init__(self, axes, functions):
        # Need the axes to "undo" the transformation of the functions in the axes
        # Have to p2c the output of the functions to get out of the axes' space
        self.axes = axes
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
                val += self.axes.p2c(f.function(x))[1]

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
                value += pow(self.axes.p2c(f.function(x))[1] - mean, 2)

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
    dots_mobj = []
    fns_mobj = []

    def __init__(self, degree, nfirst, nextra, **kwargs):
        self.nfirst = nfirst
        self.nextra = nextra
        self.degree = degree
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

        self.true_fn = degfungraph(
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
        self.play(Create(self.true_fn))
        self.play(Write(self.true_flabel))

    def fns_and_dots(self, seed):
        xs, ys, config = simple_poly_regression_get_data(seed=seed)
        fg = degfungraph(self.axes, xs, ys, self.degree, COL_PURPLE, config)
        dots = get_dots_for_axes(xs, ys, self.axes, config)

        return fg, dots

    def highlight_area_between_fn(self, f1, f2, dx=0.001, color=COL_RED, opacity=0.01):
        xs = np.arange(self.x_min + 0.015, self.x_max, dx) + (dx / 2.0)
        rects = VGroup()
        for x in xs:
            # Evaluate functions (in axes space)
            p1 = f1.function(x)
            p2 = f2.function(x)

            # Make bounding box (in axes space)
            p_x = p1[0]
            x_left = p_x - dx / 2
            x_right = p_x + dx / 2
            upper = max(p1[1], p2[1])
            lower = min(p1[1], p2[1])

            up_left = np.array([x_left, upper, 0])
            up_right = np.array([x_right, upper, 0])
            down_left = np.array([x_left, lower, 0])
            down_right = np.array([x_right, lower, 0])

            rects.add(
                Polygon(
                    up_left,
                    up_right,
                    down_right,
                    down_left,
                    color=color,
                    stroke_opacity=opacity,
                )
            )
        self.play(Create(rects))
        self.wait(0.5)

    def explain_first_function(self):
        fg, dots = self.fns_and_dots(seed=1001001)
        self.fns_mobj.append(fg)

        flabel = BMathTex("f_{\hat{w}}(x)", color=COL_PURPLE)

        # move to the right of the axes
        flabel.move_to(
            fg.function(self.x_max) + RIGHT * 0.75
        )

        self.play(Create(dots))
        self.play(Create(fg))
        self.play(Write(flabel))
        self.wait(0.5)
        self.play(fg.animate.set_color(GRAY), FadeOut(dots), FadeOut(flabel))

    def slow_draw_functions(self):
        for i in range(1, self.nfirst):
            fg, dots = self.fns_and_dots(seed=1001001 + 147 * i)
            self.fns_mobj.append(fg)
            self.play(Create(dots), run_time=0.5)
            self.play(Create(fg), run_time=0.5)
            self.play(fg.animate.set_color(GRAY), FadeOut(dots))

    def fast_draw_functions(self):
        for i in range(self.nextra):
            fg, dots = self.fns_and_dots(seed=1101001 + 147 * i)
            fg.set_color(GRAY)
            dots.set_color(GRAY)
            self.dots_mobj.append(dots)
            self.fns_mobj.append(fg)
            self.play(Create(dots), run_time=0.1)
            self.play(Create(fg), run_time=0.1)
            self.play(FadeOut(dots), run_time=0.1)

    def remove_dots_and_fns(self):
        self.play(FadeOut(VGroup(*self.fns_mobj)), run_time=1.0)

    def draw_mean_function(self):
        if not self._function_set:
            self._function_set = FunctionSet(self.axes, self.fns_mobj)

        self.meanf = self.axes.plot(self._function_set.mean, x_range=(self.x_min, self.x_max), color=COL_RED)
        meanf_label = BMathTex(r"\overline{f_{\hat{w}}}(x)", color=COL_RED)
        meanf_label.move_to(self.meanf.function(self.x_max) + RIGHT * 0.75, aligned_edge=UP)

        self.play(Create(self.meanf))
        self.play(Write(meanf_label))

    def draw_variance_interval(self):
        if not self._function_set:
            self._function_set = FunctionSet(self.axes, self.fns_mobj)

        # want to draw upper and lower function bounds for the variance
        self.upper_varf = self.axes.plot(self._function_set.upper_confidence_bound, x_range=(self.x_min, self.x_max), color=COL_GOLD)
        self.lower_varf = self.axes.plot(self._function_set.lower_confidence_bound, x_range=(self.x_min, self.x_max), color=COL_GOLD)

        print(self._function_set._lower_confidence)
        print(self._function_set._upper_confidence)

        var_label = BMathTex(
            r"\overline{\left(\overline{f_{\hat{w}}} - f_{\hat{w}}\right)^2}",
            color=COL_GOLD,
        )
        var_label.move_to(self.axes.c2p(self.x_max, self.y_min - 0.475, 0) + RIGHT * 0.75, aligned_edge=LEFT)

        self.play(
            Create(self.upper_varf),
            Create(self.lower_varf),
            Write(var_label),
        )
