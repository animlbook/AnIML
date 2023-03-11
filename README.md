# CSE/STAT 416: Introduction to Machine Learning

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://hschafer.github.io/Intro-ML/)

Author: [Hunter Schafer](https://homes.cs.washington.edu/~hschafer/)

This repository maintains the source code for our Java Tutorial. This tutorial is designed for students who already have programming experience
in some text-based language (e.g., Python) but have not yet experienced Java.

## Feedback or Spot a Bug?

If you have any feedback about the book text or structure, or you spot a bug somewhere in the book, please let us know! The best way to contact us
is to make an [GitHub Issue](https://github.com/hschafer/Intro-ML/issues) or to contact [Hunter Schafer](https://homes.cs.washington.edu/~hschafer/) directly.

## Contributing

This book is built with the [Sphinx Book Theme](https://sphinx-book-theme.readthedocs.io/en/latest/index.html) to generate HTML.

### Setup

Confusingly, we have a separate set of dependenceis to build the animations. The reason these are not part of the main setup is that they currently don't work on our CI build. So we first have to develop the animations locally and then commit the video files before building a new version of the book.

### Install Publishing Build Dependencies

Create a virtual environment with Python 3.9 or higher. For example, if you use Anaconda you can write:

```bash
conda create --name 416-book python=3.9
conda activate 416-book
```

Install the book theme dependencies. 

All of these are libraries used for themes/templating in the book. `Sphinx` is the documentation templating tool, `sphinx-book-theme` is the specific book theme, `myst-nb` changes the Sphinx langauge from rST to MyST (more similar to Markdown), and `sphinx-thebe` allows interactive notebooks in the browser. 

```bash
pip install -r requirements.txt
```

### Install Develepor Dependencies

1. To install the library for generating animations, follow the instructions [here](https://docs.manim.community/en/stable/installation.html).
2. Also install LaTeX on your system with whatever method is best.

### Editing the book

The book text is stored in `book_source/source`. Each MyST file (`.md`) corresponds to a single page of the book. Some pages, like the `index.md` files for the Modules don't contain any useful information other than links. Some of the book pages are Juptyer notebooks which also get converted to HTML.

Edit the book text by editing the appropriate MyST file. See [MyST's documentation](https://myst-parser.readthedocs.io/en/latest/) for syntax examples (note: it is incredibly similar to plain markdown, with some extra macros available).

The practice problem starter code and tests live in `book_source/coding_problems`.

### Rebuilding the book

Build the new book HTML by running:

```
# From the top-most directory
jupyter-book build book_source/source

# Or with the make command
make html
```

This will rebuild the whole book into the `book_source/source/_build` directory, which might take some time depending on the change.

### Committing and pushing changes

Stage any changes to the `book_source` and push. We **do not** stage any changes to build files. Whenever we push to `main`,
GitHub Actions will build the site again and deploy it to the `gh-pages` branch.

**Special note aboute deploying:**

This will likely not matter, but is a bug we ran into a few times when setting up the book so I thought we should docunment it. T
here must be a file called `.nojekyll` in the directory wherever GitHub Pages is deployed. This file exists on the `gh-pages` branch
and should stay there by itself. If something weird happens though, check to make sure it is still there.
