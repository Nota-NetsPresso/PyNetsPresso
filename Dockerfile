FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG TENSORFLOW_VERSION="2.8.0"
ARG PROTOBUF_VERSION="3.20.2"

RUN apt-get update && \
    apt-get install -y \
    git \ 
    vim \
    curl \
    zip \ 
    unzip \ 
    wget \
    htop \
    ncdu \
    tmux \
    screen \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir tensorflow-gpu==${TENSORFLOW_VERSION} protobuf==${PROTOBUF_VERSION} && rm -rf /root/.cache/pip

# set environment variables
ENV HOME=/app
ENV APP_PATH=$HOME/pynetspresso

# locale settings are needed for python uvicorn compatibility
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH $APP_PATH
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR $APP_PATH

# copy files to docker internal
COPY . $APP_PATH/

RUN pip install -r requirements.txt && rm -rf /root/.cache/pip
