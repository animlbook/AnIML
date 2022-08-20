SRC_DIR=book_source/source

html:
	jupyter-book build "$(SRC_DIR)"

animations:
	python genvids.py --copy manim_animations

animations_hard:
	python genvids.py --hard --copy manim_animations

clean:
	jupyter-book clean "$(SRC_DIR)"

html-deploy: animations
	jupyter-book build -W "$(SRC_DIR)"

