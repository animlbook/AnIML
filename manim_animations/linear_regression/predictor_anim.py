from lr_utils import *


class Animation(LinearScene):
    def construct(self):
        self.custom_setup(line_color=BLUE)
        self.add(self.graph, self.text_group, self.dots)

        # Draw the line and the text
        self.play(Write(self.predictor_text))
        self.wait(1)
        self.play(Write(self.line))
        self.wait(2)

        # Highlight error for one point
        residuals = VGroup()
        for x, y in zip(XS, YS):
            # index = 3 # 10
            point = self.axes.coords_to_point(x, y, 0)
            prediction = self.w_0 + self.w_1 * x
            prediction = self.axes.coords_to_point(x, prediction, 0)
            residual = DashedLine(start=prediction, end=point, color=RED)
            residuals.add(residual)

        # Text for residual
        residual_text = BText(r"Errors observed", font_size=40, color=RED)
        residual_text.scale(TEXT_SCALE)
        residual_text.to_corner(UP + RIGHT,  buff=0.25)
        residual_explanation_text = BMathTex(r"y_i - \hat{f}(x_i)",
            tex_to_color_map={"\hat{f}(": BLUE, ")": BLUE})
        residual_explanation_text.next_to(residual_text, DOWN, 0.1)
        residual_explanation_text.scale(TEXT_SCALE)

        # Animate text and residual
        self.play(Write(residual_text), Write(residual_explanation_text))
        self.wait()
        self.add(residuals, self.function, self.line, self.dots)  # Order matters
        self.play(Create(residuals), duration=4)
        self.wait(3)
