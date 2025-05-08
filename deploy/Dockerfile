ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=jammy-20250415.1

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages for setting up python 3.10
RUN apt update && \
    apt install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y python3.11 && \
    apt-get clean

# Install pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN rm -rf /var/lib/apt/lists/*

# Uninstall build packages
RUN apt autoremove -y curl software-properties-common

# Download the sources of apt packages within the container for standard legal compliance
RUN sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list
RUN apt update
# xz-utils is needed to pull the source and unpack them correctly
RUN apt install xz-utils -y
RUN mkdir -p /legal/source
WORKDIR /legal/source
# Read installed packages, strip all but the package names, pipe to 'apt source' to download respective packages
RUN apt list --installed |  grep -i installed | sed 's|\(.*\)/.*|\1|' | xargs apt source --download-only
# The source is saved in directories as well as tarballs in the current dir
RUN rm xz-utils*
COPY LICENSE-3rd-party.txt /legal/

# Install uv
RUN pip install uv

# Copy dependency file
COPY uv.lock ./
COPY pyproject.toml ./

# Set default port
ENV PORT=8000

# RRA commented out
# RUN --mount=type=bind,source=src/agent/requirements.txt,target=/opt/agent/requirements.txt \
#     pip3 install --no-cache-dir -r /opt/agent/requirements.txt

# Development target
FROM base as dev
RUN uv sync
RUN uv pip freeze > requirements.txt
COPY src/ ./src/
COPY config/ ./config/
CMD ["sh", "-c", "uv run uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]

# Production target -- default target
FROM base as prod
RUN uv sync --no-dev
RUN uv pip freeze > requirements.txt
COPY src/ ./src/
COPY config/ ./config/
CMD ["sh", "-c", "uv run uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
