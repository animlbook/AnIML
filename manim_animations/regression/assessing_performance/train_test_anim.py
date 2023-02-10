from train_test_scene import *


class Animation(BTrainTestScene):
    def construct(self):
        self.setup_scene()

        # show optimimum position
        opt_line = DashedLine(
            self.axes.c2p(5, self.true_error(5), 0),
            self.axes.c2p(5, 0, 0),
            color=COL_PURPLE,
        )
        opt_label = BMathTex(r"p^{*}", color=COL_PURPLE)
        opt_label.next_to(opt_line, DOWN)

        self.play(Create(opt_line))
        self.play(Write(opt_label))

        x_min, x_max = X_RANGE
        y_min, y_max = Y_RANGE

        underfit_rect = Polygon(
            self.axes.c2p(1.0, y_max, 0),
            self.axes.c2p(3.5, y_max, 0),
            self.axes.c2p(3.5, y_min, 0),
            self.axes.c2p(1.0, y_min, 0),
            stroke_opacity=0.0,
            fill_color=COL_RED,
            fill_opacity=0.1,
        )

        underfit_text = BTex(r"Underfit \\ Model", color=COL_BLACK)
        underfit_text.scale(0.75)
        underfit_text.next_to(underfit_rect, UP)

        self.play(FadeIn(underfit_rect))
        self.play(Write(underfit_text))

        overfit_rect = Polygon(
            self.axes.c2p(7.5, y_max, 0),
            self.axes.c2p(x_max, y_max, 0),
            self.axes.c2p(x_max, y_min, 0),
            self.axes.c2p(7.5, y_min, 0),
            stroke_opacity=0.0,
            fill_color=COL_RED,
            fill_opacity=0.1,
        )

        overfit_text = BTex(r"Overfit \\ Model", color=COL_BLACK)
        overfit_text.scale(0.75)
        overfit_text.move_to(self.axes.c2p(11, 3, 0))
        overfit_text.next_to(overfit_rect, UP)

        self.play(FadeIn(overfit_rect))
        self.play(Write(overfit_text))
