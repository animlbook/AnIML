FROM mcr.microsoft.com/devcontainers/python:3.9

# Kind of gross, but copied this from Manim's Dockerfile to build manim
# https://github.com/ManimCommunity/manim/blob/main/docker/Dockerfile
# Can't use their image since I need 3.9 :(

RUN apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        build-essential \
        gcc \
        cmake \
        libcairo2-dev \
        libffi-dev \
        libpango1.0-dev \
        freeglut3-dev \
        pkg-config \
        make \
        wget \
        ghostscript 

# setup a minimal texlive installation
COPY .devcontainer/texlive-profile.txt /tmp/
ENV PATH=/usr/local/texlive/bin/armhf-linux:/usr/local/texlive/bin/aarch64-linux:/usr/local/texlive/bin/x86_64-linux:$PATH
RUN wget -O /tmp/install-tl-unx.tar.gz http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz && \
    mkdir /tmp/install-tl && \
    tar -xzf /tmp/install-tl-unx.tar.gz -C /tmp/install-tl --strip-components=1 && \
    /tmp/install-tl/install-tl --profile=/tmp/texlive-profile.txt \
    && tlmgr install \
        amsmath babel-english cbfonts-fd cm-super ctex doublestroke dvisvgm everysel \
        fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin \
        mathastext microtype ms physics preview ragged2e relsize rsfs \
        setspace standalone tipa wasy wasysym xcolor xetex xkeyval

## End Manim Dockerfile

RUN apt-get install graphviz

## Set up Env
# ENV PATH="$HOME/.local/bin:$PATH"

# Run pip for the rest
COPY requirements.txt .devcontainer/noop.txt /tmp/
RUN pip install -r /tmp/requirements.txt
