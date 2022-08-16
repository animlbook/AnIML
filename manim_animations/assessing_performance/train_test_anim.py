from train_test_scene import *


class Animation(BTrainTestScene):
    def construct(self):
        self.setup_scene()

        # show optimimum position
        opt_line = DashedLine(
            np.array([5, self.true_fn.function(5), 0]),
            np.array([5, 0, 0]),
            color=COL_BLUE,
        )
        opt_label = BTex(r"p^{*}", color=COL_BLUE)
        opt_label.move_to(np.array([5.0, -0.5, 0]))
        opt_grp = VGroup(opt_line, opt_label)
        opt_grp.shift(self.centershift)
        self.play(ShowCreation(opt_line))
        self.play(Write(opt_label))

        underfit_rect = Polygon(
            np.array([0.0, self.y_max, 0]),
            np.array([3.5, self.y_max, 0]),
            np.array([3.5, self.y_min, 0]),
            np.array([0.0, self.y_min, 0]),
            stroke_opacity=0.0,
            fill_color=COL_RED,
            fill_opacity=0.1,
        )

        underfit_rect.shift(self.centershift)
        underfit_text = BText(r"Underfit \\ Model", color=COL_BLACK)
        underfit_text.scale(0.75)
        underfit_text.move_to(self.axes.c2p(1.5, 1, 0))

        self.play(FadeIn(underfit_rect))
        self.play(Write(underfit_text))

        overfit_rect = Polygon(
            np.array([7.5, self.y_max, 0]),
            np.array([self.x_max, self.y_max, 0]),
            np.array([self.x_max, self.y_min, 0]),
            np.array([7.5, self.y_min, 0]),
            stroke_opacity=0.0,
            fill_color=COL_RED,
            fill_opacity=0.1,
        )

        overfit_rect.shift(self.centershift)
        overfit_text = BText(r"Overfit \\ Model", color=COL_BLACK)
        overfit_text.scale(0.75)
        overfit_text.move_to(self.axes.c2p(11, 3, 0))

        self.play(FadeIn(overfit_rect))
        self.play(Write(overfit_text))
