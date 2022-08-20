from train_test_scene import *


def filled_rect(color, lower_left, upper_right, opacity_left, opacity_right, dx=0.005):
    xs = np.arange(lower_left[0], upper_right[0], dx)
    yl, yu = lower_left[1], upper_right[1]
    rects = []
    for i, x in enumerate(xs):
        opacity = (1.0 - float(i / len(xs))) * opacity_left + (
            float(i / len(xs))
        ) * opacity_right
        ul, ur = np.array([x, yu, 0]), np.array([x + dx, yu, 0])
        dl, dr = np.array([x, yl, 0]), np.array([x + dx, yl, 0])
        rects.append(
            Polygon(
                ul,
                ur,
                dl,
                dr,
                fill_color=color,
                fill_opacity=opacity,
                stroke_color=color,
                stroke_opacity=opacity,
            )
        )
    return VGroup(*rects)


class Animation(BTrainTestScene):
    def construct(self):
        self.setup_scene()

        x1, x2 = 6, 8
        fhat_line = DashedLine(
            np.array([x2, self.true_fn.function(x2), 0]),
            np.array([x2, 0, 0]),
            color=COL_BLACK,
        )
        fhat_label = BTex(r"\hat{f}", color=COL_BLACK)
        fhat_label.move_to(np.array([x2, -0.5, 0]))
        fhat_grp = VGroup(fhat_line, fhat_label)
        fhat_grp.shift(self.centershift)
        self.play(ShowCreation(fhat_line))
        self.play(Write(fhat_label))

        fprime_line = DashedLine(
            np.array([x1, self.true_fn.function(x1), 0]),
            np.array([x1, 0, 0]),
            color=COL_BLACK,
        )
        fprime_label = BTex(r"f'", color=COL_BLACK)
        fprime_label.move_to(np.array([x1, -0.5, 0]))
        fprime_grp = VGroup(fprime_line, fprime_label)
        fprime_grp.shift(self.centershift)
        self.play(ShowCreation(fprime_line))
        self.play(Write(fprime_label))

        # # zoom in:
        # old_view = VGroup(self.axes_and_fn_label, fprime_grp, fhat_grp)
        # new_view = old_view.deepcopy()
        # zoom_scale = 3.0
        # new_view.shift(zoom_scale * -self.axes.c2p(7, 1.75, 0))
        # new_view.scale(zoom_scale)
        # self.play(Transform(old_view, new_view))

        # # draw brace
        # x = 8.1
        # grp = VGroup(Dot(self.axes.c2p(x, self.true_fn.function(x2), 0), color=COL_BLACK),
        #              Dot(self.axes.c2p(x, self.true_fn.function(x1), 0), color=COL_BLACK))
        # b1 = Brace(grp, RIGHT, color=COL_BLACK)
        # self.play(ShowCreation(b1))
