from manim_config import *

from config import *

XS = np.array([1, 2, 4])
YS = np.array([1, 2, 4])[np.newaxis].T
config = {
    "X_MIN": 0,
    "X_MAX": 6,
    "Y_MIN": 0,
    "Y_MAX": 6,
}


class Animation(BScene):
    def construct(self):
        axes, dots = axes_and_data(XS, YS, config)

        plot = VGroup(axes, dots)
        label = BText("Given two points, which polynomial model fits best?")
        label.next_to(plot, UP)

        deg1 = degfungraph(XS, YS, 1, COL_RED, config)
        deg1label = BText("$p = 1$?")
        deg1label.next_to(plot, UP)
        deg1grp = VGroup(deg1, deg1label)

        deg2 = degfungraph(XS, YS, 2, COL_GREEN, config)
        deg2label = BText("$p = 2$?")
        deg2label.next_to(plot, UP)
        deg2grp = VGroup(deg2, deg2label)

        begingrp = VGroup(plot, label)
        grp = VGroup(begingrp, deg1grp, deg2grp)
        grp.move_to((0, 0, 0))

        self.play(ShowCreation(plot), Write(label))
        self.play(Transform(begingrp, deg1grp))
        self.wait(1)
        self.play(Transform(deg1grp, deg2grp))
