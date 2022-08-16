from manim_config import *
from config import *


GRAPH_CONFIG = {
    "X_MIN": 0,
    "X_MAX": 12,
    "Y_MIN": 0,
    "Y_MAX": 6,
}


class FunctionOffGraph(FunctionGraph):
    def __init__(self, function=None, **kwargs):
        self.y_min, self.y_max = 0, 1
        if "y_min" in kwargs:
            self.y_min = kwargs["y_max"]
        if "y_max" in kwargs:
            self.y_max = kwargs["y_max"]

        super().__init__(function=function, **kwargs)

    def get_y_bound(self, lp):
        if lp[1] > ((self.y_max - self.y_min) / 2):
            return self.y_max
        else:
            return self.y_min

    def generate_points(self):
        t_min, t_max = self.t_min, self.t_max
        dt = self.dt

        discontinuities = filter(lambda t: t_min <= t <= t_max, self.discontinuities)
        discontinuities = np.array(list(discontinuities))
        boundary_times = [
            self.t_min,
            self.t_max,
            *(discontinuities - dt),
            *(discontinuities + dt),
        ]
        boundary_times.sort()
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            step_size = self.get_step_size(t1)
            t_range = list(np.arange(t1, t2, step_size))
            if t_range[-1] != t2:
                t_range.append(t2)
            points = np.array([self.function(t) for t in t_range])
            cur_draw = []
            for p in points:
                if p[1] > self.y_min and p[1] < self.y_max:
                    # add an interpolation point if we're just starting a new line
                    if len(cur_draw) == 0 and not (
                        np.isclose(p[0], self.x_min) or np.isclose(p[1], self.x_max)
                    ):
                        cur_draw.append(
                            np.array([p[0] - step_size, self.get_y_bound(p), p[2]])
                        )
                    cur_draw.append(p)
                elif len(cur_draw) > 0:
                    # if the last point was close to the top, then use the
                    # top as the last point, and vice/versa with the min
                    lp = cur_draw[-1]
                    cur_draw.append(np.array([p[0], self.get_y_bound(lp), p[2]]))
                    self.start_new_path(cur_draw[0])
                    self.add_points_as_corners(cur_draw[1:])
                    cur_draw = []
            if len(cur_draw) > 0:
                self.start_new_path(cur_draw[0])
                self.add_points_as_corners(cur_draw[1:])
        self.make_smooth()
        return self


def simple_poly_regression_true_data():
    dim = 5
    XTRUE = np.array([1.0, 3.0, 6.0, 8.1, 10.5, 11.9])
    YTRUE = np.array([1.8, 4.5, 1.75, 3.0, 2.5, 3.0])

    return dim, XTRUE, YTRUE[np.newaxis].T, GRAPH_CONFIG


def simple_poly_regression_get_data(scale=1.0, seed=100394, N=25):
    dim, XTRUE, YTRUE, config = simple_poly_regression_true_data()

    b = beta(H(XTRUE, dim), YTRUE)

    np.random.seed(seed)
    XS = np.random.uniform(config["X_MIN"], config["X_MAX"], N)
    YS = np.zeros(N)
    valididx = [True] * N
    for i, x in enumerate(XS):
        YS[i] = fhat(x, XTRUE, YTRUE, dim) + np.random.normal(0, 0.6)
        if YS[i] < config["Y_MIN"] or YS[i] > config["Y_MAX"]:
            valididx[i] = False

    XS = XS[valididx]
    YS = YS[valididx]

    config["X_MIN"] *= scale
    config["X_MAX"] *= scale
    config["Y_MIN"] *= scale
    config["Y_MAX"] *= scale

    return scale * XS, scale * YS[np.newaxis].T, config


def h(x, deg):
    pows = np.zeros(deg + 1)
    for i in range(deg + 1):
        pows[i] = pow(x, i)
    return pows


def H(Xtrain, deg):
    N = Xtrain.shape[0]
    X = np.empty((N, deg + 1))
    for i in range(N):
        X[i] = h(Xtrain[i], deg)
    return X


def beta(HX, Y):
    assert Y.shape[1] == 1
    return np.matmul(np.linalg.inv(np.matmul(HX.T, HX)), np.matmul(HX.T, Y))


def fhat(x, Xtrain, Ytrain, deg):
    xhat = h(x, deg)
    b = beta(H(Xtrain, deg), Ytrain)
    return np.matmul(xhat, b)[0]  # the matmul returns a 1x1 array


def fhat_vector(X, Xtrain, Ytrain, deg):
    xhat = H(X, deg)
    b = beta(H(Xtrain, deg), Ytrain)
    return np.matmul(xhat, b)  # the matmul returns a 1x1 array


def RSS(y, beta, x):
    return np.linalg.norm(y - np.matmul(fhat(x, beta)))


def get_dots_for_axes(XS, YS, axes, config, radius=DEFAULT_DOT_RADIUS):
    xmin, xmax = config["X_MIN"], config["X_MAX"]
    ymin, ymax = config["Y_MIN"], config["Y_MAX"]

    # Draw points
    dots = VGroup()
    for x, y in zip(XS, YS):
        if x > xmin and x < xmax and y > ymin and y < ymax:
            point = axes.coords_to_point(x, y, 0)
            dot = Dot(point, color=COL_BLACK, radius=radius)
            dots.add(dot)

    return dots


def axes_and_data(XS, YS, config, pos=(0.0, 0.0, 0.0), radius=DEFAULT_DOT_RADIUS):
    xmin, xmax = config["X_MIN"], config["X_MAX"]
    ymin, ymax = config["Y_MIN"], config["Y_MAX"]
    axes = Axes(
        x_min=xmin,
        x_max=xmax,
        y_min=ymin,
        y_max=ymax,
        center_point=pos,
        axis_config={"include_tip": False, "include_ticks": False, "color": GREY_C},
    )

    dots = get_dots_for_axes(XS, YS, axes, config)
    return axes, dots


def degfungraph(Xtrain, Ytrain, deg, color, config):
    assert Ytrain.shape[1] == 1
    assert Ytrain.shape[0] == len(Xtrain)

    def f(x):
        yhat = fhat(x, Xtrain, Ytrain, deg)
        return yhat

    return FunctionOffGraph(
        x_min=config["X_MIN"],
        x_max=config["X_MAX"],
        y_min=config["Y_MIN"],
        y_max=config["Y_MAX"],
        function=f,
        color=color,
    )
