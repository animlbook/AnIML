from lr_utils import *


class Animation(GradientDescentScene):
    def construct(self):
        # Function to draw
        def f(x):
            return (x - 2) ** 2

        super().custom_setup(f, 0.7, 2)
