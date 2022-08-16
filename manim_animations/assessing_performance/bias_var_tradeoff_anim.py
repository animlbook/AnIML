from train_test_scene import *


def noise(x):
    return 0.75


def bias(x):
    return np.exp(-x / 5.0 + 1)


def var(x):
    return np.exp(x / 5.0 - 1)


def true_error(x):
    return bias(x) ** 2 + var(x) + noise(x)


class Animation(BTrainTestScene):
    def function_and_label(self, fn, text, color):
        fng = FunctionOffGraph(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            function=fn,
            color=color,
        )

        fnl = BText(text, color=color)
        fnl.scale(0.75)
        ypos = max(self.y_min, min(self.y_max, fng.function(self.x_max)))
        fnl.move_to(self.axes.c2p(self.x_max, ypos, 0) + RIGHT * 0.75)

        return VGroup(fng, fnl)

    def construct(self):
        self.setup_axes()

        bias_grp = self.function_and_label(bias, "Bias", COL_BLUE)
        var_grp = self.function_and_label(var, "Variance", COL_GREEN)
        noise_grp = self.function_and_label(noise, "Noise", COL_PURPLE)
        true_grp = self.function_and_label(true_error, r"True \\ Error", COL_GOLD)

        title = BText(
            "True Error = Bias$^2$ + Variance + Noise",
            tex_to_color_map={
                "True Error": COL_GOLD,
                "Bias$^2$": COL_BLUE,
                "Noise": COL_PURPLE,
                "Variance": COL_GREEN,
            },
        )
        title.scale(0.75)
        title.next_to(self.axes, UP)

        self.axes_and_fn_label = VGroup(
            self.axes, bias_grp, var_grp, noise_grp, true_grp, title, *self.axes_labels
        )
        self.centershift = -self.axes_and_fn_label.get_center()
        self.axes_and_fn_label.move_to((0, 0, 0))

        self.play(
            ShowCreation(self.axes), Write(VGroup(*self.axes_labels)), Write(title)
        )
        self.play(ShowCreation(bias_grp))
        self.play(ShowCreation(var_grp))
        self.play(ShowCreation(noise_grp))
        self.play(ShowCreation(true_grp))
