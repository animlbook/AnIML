from lr_utils import *


class Animation(LinearScene):
    def construct(self):
        self.custom_setup(line_color=BLUE)
        self.add(self.graph, self.text_group, self.dots)

        # Draw the line and the text
        self.play(Write(self.predictor_text))
        self.wait(1)
        self.play(ShowCreation(self.line))
        self.wait(2)

        # Highlight error for one point
        residuals = Group()
        for x, y in zip(XS, YS):
            # index = 3 # 10
            point = self.axes.coords_to_point(x, y, 0)
            prediction = self.w_0 + self.w_1 * x
            prediction = self.axes.coords_to_point(x, prediction, 0)
            residual = DashedLine(start=prediction, end=point, color=RED)
            residuals.add(residual)

        # Text for residual
        residual_text = BText(r"Errors observed", font_size=30, color=RED)
        residual_text.next_to(self.graph, UP + RIGHT, buff=0.5)

        # Animate text and residual
        self.play(Write(residual_text))
        self.wait()
        self.add(residuals, self.function, self.line, self.dots)  # Order matters
        self.play(ShowCreation(residuals), duration=4)
        self.wait(3)
