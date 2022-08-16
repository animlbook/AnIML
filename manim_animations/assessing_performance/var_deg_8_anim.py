from manim_config import *
from bias_var_scene import BBiasVarianceScene


class Animation(BBiasVarianceScene):
    def __init__(self, **kwargs):
        nfirst = 3
        nextra = 20
        super().__init__(8, nfirst, nextra, **kwargs)

    def construct(self):
        self.setup_scene()
        self.explain_first_function()
        self.slow_draw_functions()
        self.fast_draw_functions()
        self.draw_mean_function()
        self.draw_variance_interval()
        self.remove_dots_and_fns()
        self.highlight_area_between_fn(self.lower_varf, self.upper_varf, color=COL_GOLD)
