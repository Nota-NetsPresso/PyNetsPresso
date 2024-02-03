FROM python:3.8.16

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

RUN mkdir -p /home/appuser/netspresso
WORKDIR /home/appuser/netspresso

COPY . /home/appuser/netspresso

RUN pip install -r requirements.txt && rm -rf /root/.cache/pip
