from lr_utils import *


class Animation(LinearScene):
    def construct(self):
        self.custom_setup(line_color=BLUE)
        self.add(self.axes, self.dots)

        coeffs = [(2, 0.5), (3, 0.1), (1, 0.4), (2.8, -0.02), (3.5, -0.5), (1.5, 0.2)]

        lines = Group()
        for w_0, w_1 in coeffs:
            line = self.linear_function(w_0, w_1, line_color=BLUE)
            lines.add(line)

        self.add(lines, self.dots)
        self.play(ShowCreation(lines, run_time=5, lag_ratio=1))
        self.wait(2)
