FROM manimcommunity/manim:stable

# Set up Env
ENV PATH="$HOME/.local/bin:$PATH"

# Run pip for the rest
COPY requirements.txt .devcontainer/noop.txt /tmp/
RUN pip install -r /tmp/requirements.txt
