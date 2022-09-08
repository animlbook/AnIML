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

html: animations images
	jupyter-book build "$(SRC_DIR)"

html-deploy: animations_hard images_hard
	jupyter-book build "$(SRC_DIR)"

