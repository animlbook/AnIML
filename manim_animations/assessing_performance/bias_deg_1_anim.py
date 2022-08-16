from bias_var_scene import BBiasVarianceScene


class Animation(BBiasVarianceScene):
    def __init__(self, **kwargs):
        nfirst = 3
        nextra = 3
        super().__init__(1, nfirst, nextra, **kwargs)

    def construct(self):
        self.setup_scene()
        self.explain_first_function()
        self.slow_draw_functions()
        self.fast_draw_functions()
        self.draw_mean_function()
        self.remove(*self.fns_mobj)
        self.highlight_area_between_fn(self.meanf, self.true_fn)
