import colour
from manim import *

COL_BACKGROUND = "#ffffff" #fffff8"
COL_BLACK = "#111111"
COL_GREEN = GREEN_E
COL_PURPLE = PURPLE
COL_RED = RED_E
COL_BLUE = BLUE_D
COL_GOLD = "#ab7a22"

TEXT_SCALE = 0.6


def text_kwargs(kwargs):
    text_kwargs = {
        "color": COL_BLACK,
        "background_stroke_width": 0.0,
        "stroke_width": 1.5,
    }
    text_kwargs.update(**kwargs)
    return text_kwargs


class BScene(Scene):
    def __init__(self, **kwargs):
        config["background_color"] = colour.Color(COL_BACKGROUND)
        super().__init__(**kwargs)


class BTex(Tex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **text_kwargs(kwargs))


class BMathTex(MathTex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **text_kwargs(kwargs))


class BText(Text):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **text_kwargs(kwargs))


def make_box(
    text_string,
    text_color=COL_BLACK,
    bg_color=COL_GREEN,
    bg_fill_color=COL_GREEN,
    fill_opacity=0.7,
):
    label = BText(text_string, color=text_color)
    bg = Rectangle(color=bg_color, fill_color=bg_fill_color, fill_opacity=fill_opacity)
    bg.surround(label, buff=1.0)

    return VGroup(bg, label)


# when you want to write some text to be transformed centered at a previous
# text object
def replacement_text(prev, text, **kwargs):
    text_mobj = BText(text, **kwargs)
    text_mobj.move_to(prev.get_center())

    return text_mobj
