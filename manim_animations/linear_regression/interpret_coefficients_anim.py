from lr_utils import *


class Animation(LinearScene):
    def construct(self):
        TEXT_SCALE = 0.7

        self.custom_setup(line_color=GREEN)
        self.add(self.axes, self.dots)

        # Model text
        model_text = BTex(
            r"\text{Model:}\ y_i = w_0 + w_1x_i + \varepsilon_i",
            color=GREEN,
            tex_to_color_map={r"Model:": COL_BLACK},
        )
        model_text.to_corner(UP + LEFT)

        self.play(Write(model_text))
        self.add(self.line, self.dots)  # To draw below
        self.play(ShowCreation(self.line))

        # Highlight w_0
        w_0_point = self.axes.coords_to_point(0, self.w_0, 0)
        w_0_point = Dot(w_0_point, color=GREEN)
        w_0_text = BTex(r"w_0", color=GREEN)
        w_0_text.scale(TEXT_SCALE)
        w_0_text.next_to(w_0_point, LEFT)
        self.play(Write(w_0_text), FadeIn(w_0_point))

        # Highlight slope
        def eval(x):
            return self.w_0 + self.w_1 * x

        start = 1.15
        end = 2.62  # 1.9
        start_point = self.axes.coords_to_point(start, eval(start), 0)
        end_point = self.axes.coords_to_point(end, eval(end), 0)

        corner = self.axes.coords_to_point(end, eval(start), 0)

        # Create run
        run = DashedLine(start=start_point, end=corner, color=BLUE)
        run_text = BTex(r"1", color=BLUE)
        run_text.scale(TEXT_SCALE)
        run_text.next_to(run, DOWN)
        self.play(ShowCreation(run), FadeIn(run_text))

        # Create rise
        rise = DashedLine(start=corner, end=end_point, color=BLUE)
        rise_text = BTex(r"w_1", color=BLUE)
        rise_text.scale(TEXT_SCALE)
        rise_text.next_to(rise, RIGHT, buff=0.05)
        self.play(ShowCreation(rise), FadeIn(rise_text))

        self.wait(3)
