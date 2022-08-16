from lr_utils import *


class Animation(ModelScene):
    def construct(self):
        TEXT_SCALE = 0.7
        TEXT_LBUFF = 0.4

        # Generate the axes
        self.custom_setup()
        self.add(self.graph, self.text_group, self.dots)

        # Write Model
        model_text = BTex(
            r"\text{Model:}\ y_i = f(x_i) + \varepsilon_i",
            tex_to_color_map={
                "f(": GREEN,
                ")": GREEN,
                "y_i": BLUE,
                r"\varepsilon_i": MAROON,
            },
        )

        model_text.to_corner(UP + RIGHT)
        self.play(Write(model_text))

        # Pick one point to highlight
        index = 19
        origin = self.axes.coords_to_point(0, 0, 0)
        point = self.axes.coords_to_point(XS[index], YS[index], 0)
        expected_value = self.axes.coords_to_point(XS[index], f(XS[index]), 0)

        # Draw a vertical line from its x to the expected value
        x_mark = Line(
            start=[point[0], origin[1], 0],
            end=[point[0], origin[1] + 0.1, 0],
            color=COL_BLACK,
        )
        x_text = BTex(r"x_7")
        x_text.next_to(x_mark, direction=DOWN)
        x_text.scale(TEXT_SCALE)

        dot = Dot(expected_value, color=GREEN)
        ev_line = DashedLine(
            start=[point[0], origin[1], 0],
            end=[point[0], expected_value[1], 0],
            color=GREEN,
        )

        self.play(
            AnimationGroup(FadeIn(x_mark), FadeIn(x_text)),
            FadeIn(ev_line, duration=2),
            FadeIn(dot),
            lag_ration=0.9,
            duration=4,
        )
        self.wait()

        ev_text = BTex(r"f(x_i)", tex_to_color_map={"f(": GREEN, ")": GREEN,})
        self.play(
            ApplyMethod(
                ev_line.next_to, origin - np.array([0.05, 0, 0]), UP, {"buff": 0}
            )
        )
        ev_text.next_to(ev_line, direction=LEFT * TEXT_LBUFF)
        ev_text.scale(TEXT_SCALE)
        self.play(FadeIn(ev_text))

        # Draw the epislon from the expected to observed
        epsilon = DashedLine(start=expected_value, end=point, color=MAROON)
        epsilon_text = BTex(r"\epsilon_7", color=MAROON)
        epsilon_text.scale(TEXT_SCALE)
        epsilon_text.next_to(epsilon, direction=LEFT * TEXT_LBUFF)
        self.play(ShowCreation(epsilon, duration=1), Write(epsilon_text))
        self.wait()

        # Draw a line to show how it computes the y from that
        y_line = DashedLine(
            start=[point[0], origin[1], 0], end=[point[0], point[1], 0], color=BLUE
        )
        y_text = BTex(r"y_7", color=BLUE)
        y_text.next_to(y_line, direction=LEFT * TEXT_LBUFF)
        y_text.scale(TEXT_SCALE)
        self.play(FadeIn(y_line, duration=2), Write(y_text), lag_ratio=0.5)
        self.wait(duration=4)
