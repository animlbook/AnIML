from manim_config import *
from config import *


class MeanFunction(FunctionOffGraph):
    def __init__(self, functions, **kwargs):
        self.functions = functions

        def func(x):
            s = 0.0
            for f in self.functions:
                s += f.function(x)
            return s / len(self.functions)

        super().__init__(function=func, **kwargs)


class VarianceFunction(FunctionOffGraph):
    def __init__(self, upper_fn, mean_function, functions, **kwargs):
        # If this is the upper bound of the variace. The symetric lower function
        # bound will use upper_fn = False
        self.upper_fn = upper_fn
        self.mean_function = mean_function
        self.functions = functions

        def func(x):
            side = 1.0 if self.upper_fn else -1.0
            s = 0.0
            mean = self.mean_function.function(x)
            for f in self.functions:
                s += pow(f.function(x) - mean, 2)
            return mean + side * (s / len(self.functions))

        super().__init__(function=func, **kwargs)


class BBiasVarianceScene(BScene):
    dots_mobj = []
    fns_mobj = []

    def __init__(self, degree, nfirst, nextra, **kwargs):
        self.nfirst = nfirst
        self.nextra = nextra
        self.degree = degree
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
        self.axes, _ = axes_and_data([], [], config)

        self.x_min = config["X_MIN"]
        self.x_max = config["X_MAX"]
        self.y_min = config["Y_MIN"]
        self.y_max = config["Y_MAX"]

        self.true_fn = degfungraph(
            self.xtrue, self.ytrue, self.true_dim, COL_BLUE, config
        )
        self.true_flabel = BTex("f(x)", color=COL_BLUE)

        # move to the right of the axes
        self.true_flabel.move_to(
            self.axes.c2p(self.x_max, self.true_fn.function(self.x_max), 0)
            + RIGHT * 0.75
        )

        axes_and_fn_label = VGroup(self.axes, self.true_fn, self.true_flabel)
        self.centershift = -axes_and_fn_label.get_center()
        axes_and_fn_label.move_to((0, 0, 0))

        self.play(ShowCreation(self.axes))
        self.play(ShowCreation(self.true_fn))
        self.play(Write(self.true_flabel))

    def fns_and_dots(self, seed):
        xs, ys, config = simple_poly_regression_get_data(seed=seed)
        fg = degfungraph(xs, ys, self.degree, COL_PURPLE, config)
        fg.shift(self.centershift)
        dots = get_dots_for_axes(xs, ys, self.axes, config)

        return fg, dots

    def highlight_area_between_fn(self, f1, f2, dx=0.001, color=COL_RED, opacity=0.01):
        xs = np.arange(self.x_min + 0.015, self.x_max, dx) + (dx / 2.0)
        rects = VGroup()
        for x in xs:
            xl, xr = x - dx / 2.0, x + dx / 2.0
            y1, y2 = f1.function(x), f2.function(x)
            upper = min(max(y1, y2), self.y_max) - 0.02
            lower = max(min(y1, y2), self.y_min) + 0.02
            up_left = np.array([xl, upper, 0])
            up_right = np.array([xr, upper, 0])
            down_left = np.array([xl, lower, 0])
            down_right = np.array([xr, lower, 0])
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
        rects.shift(self.centershift)
        self.play(ShowCreation(rects))
        self.wait(0.5)

    def explain_first_function(self):
        fg, dots = self.fns_and_dots(seed=1001001)
        self.fns_mobj.append(fg)

        flabel = BTex("f_{\hat{w}}(x)", color=COL_PURPLE)

        # move to the right of the axes
        flabel.move_to(
            self.axes.c2p(self.x_max, fg.function(self.x_max), 0) + RIGHT * 0.75
        )

        self.play(ShowCreation(dots))
        self.play(ShowCreation(fg))
        self.play(Write(flabel))
        self.wait(0.5)
        self.play(fg.set_color, (GRAY), FadeOut(dots), FadeOut(flabel))

    def slow_draw_functions(self):
        for i in range(1, self.nfirst):
            fg, dots = self.fns_and_dots(seed=1001001 + 147 * i)
            self.fns_mobj.append(fg)
            self.play(ShowCreation(dots), run_time=0.5)
            self.play(ShowCreation(fg), run_time=0.5)
            self.play(fg.set_color, (GRAY), FadeOut(dots))

    def fast_draw_functions(self):
        for i in range(self.nextra):
            fg, dots = self.fns_and_dots(seed=1101001 + 147 * i)
            fg.set_color(GRAY)
            dots.set_color(GRAY)
            self.dots_mobj.append(dots)
            self.fns_mobj.append(fg)
            self.play(ShowCreation(dots), run_time=0.1)
            self.play(ShowCreation(fg), run_time=0.1)
            self.play(FadeOut(dots), run_time=0.1)

    def remove_dots_and_fns(self):
        self.play(FadeOut(VGroup(*self.fns_mobj)), run_time=1.0)

    def draw_mean_function(self):
        self.meanf = MeanFunction(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            functions=self.fns_mobj,
            color=COL_RED,
        )
        self.meanf.shift(self.centershift)
        meanf_label = BTex(r"\overline{f_{\hat{w}}}(x)", color=COL_RED)
        y_pos = min(max(self.meanf.function(self.x_max), self.y_min), self.y_max)
        meanf_label.move_to(self.axes.c2p(self.x_max, y_pos, 0) + RIGHT * 0.75)

        self.play(ShowCreation(self.meanf))
        self.play(Write(meanf_label))

    def draw_variance_interval(self):
        # want to draw upper and lower function bounds for the variance
        self.upper_varf = VarianceFunction(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            upper_fn=True,
            mean_function=self.meanf,
            functions=self.fns_mobj,
            color=COL_GOLD,
        )
        self.lower_varf = VarianceFunction(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            upper_fn=False,
            mean_function=self.meanf,
            functions=self.fns_mobj,
            color=COL_GOLD,
        )
        self.upper_varf.shift(self.centershift)
        self.lower_varf.shift(self.centershift)

        var_label = BTex(
            r"\overline{\left(\overline{f_{\hat{w}}} - f_{\hat{w}}\right)^2}",
            color=COL_GOLD,
        )
        var_label.move_to(self.axes.c2p(self.x_max, self.y_min - 0.475, 0))

        self.play(
            ShowCreation(self.upper_varf),
            ShowCreation(self.lower_varf),
            Write(var_label),
        )
