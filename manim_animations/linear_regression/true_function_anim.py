from lr_utils import *


class Animation(ModelScene):
    def construct(self):
        # Generate the axes
        self.custom_setup()
        self.add(self.axes, self.data_text, self.dots)

        # Draw Function
        self.play(Write(self.function_text))
        self.add(self.function, self.dots)
        self.play(ShowCreation(self.function))
        self.wait(duration=1.5)
