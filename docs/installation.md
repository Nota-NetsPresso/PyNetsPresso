# Installation

## Prerequisites

- Python `3.10` | `3.11`
- PyTorch `2.0.1` (recommended) or above
- TensorFlow:
  - Python `3.10`: `2.7.0` or above (recommended: `2.8.0`)
  - Python `3.11`: `2.12.0` or above

## Install with PyPI (stable)

```bash
pip install netspresso
```

## Install with GitHub

To install with editable mode,

```bash
git clone https://github.com/nota-netspresso/pynetspresso.git
cd pynetspresso
pip install -e .
```

## Install with Docker

Please clone this repository and refer to [Dockerfile](https://github.com/Nota-NetsPresso/PyNetsPresso/blob/develop/docker-compose.yml) and [docker-compose.yml](https://github.com/Nota-NetsPresso/PyNetsPresso/blob/develop/docker-compose.yml).

### Docker with docker-compose

For the latest information, please check [docker-compose.yml](https://github.com/Nota-NetsPresso/PyNetsPresso/blob/develop/docker-compose.yml).

```bash
# run command
export TAG=v$(cat netspresso/VERSION) && \
docker compose run --service-ports --name netspresso-dev netspresso bash
```

### Docker image build

If you run with `docker run` command, follow the image build and run command in the below:

```bash
# build an image
docker build -t netspresso:v$(cat netspresso/VERSION) .
```

```bash
# docker run command
docker run -it --ipc=host\
  --gpus='"device=0,1,2,3"'\  # your GPU id(s)
  -v /PATH/TO/DATA:/DATA/PATH/IN/CONTAINER\
  -v /PATH/TO/CHECKPOINT:/CHECKPOINT/PATH/IN/CONTAINER\
  -p 50001:50001\
  -p 50002:50002\
  -p 50003:50003\
  -p 50004:50004\
  --name netspresso-dev netspresso:v$(cat netspresso/VERSION)
```
