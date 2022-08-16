from lr_utils import *


class Animation(ModelScene):
    def construct(self):
        self.custom_setup()
        self.play(Write(self.axes, run_time=2, lag_ratio=0.1))
        self.play(Write(self.data_text), FadeIn(self.dots), lag_ratio=0.1, run_time=4)
