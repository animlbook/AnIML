SRC_DIR=book_source/source


animations:
	python genvids.py --copy manim_animations

animations_hard:
	python genvids.py --hard --copy manim_animations

images:
	python genvids.py --save_last_frame --copy manim_animations

images_hard:
	python genvids.py --save_last_frame --hard --copy manim_animations

clean:
	jupyter-book clean "$(SRC_DIR)"

manim: animations images

html:
	jupyter-book build "$(SRC_DIR)"

all: manim html