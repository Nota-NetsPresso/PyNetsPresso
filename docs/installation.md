## Installation

If you want to install netspresso library, it is required to set your python version first. And choose one of installation methods described below.

### Prerequisites

- Python `3.8` | `3.9` | `3.10`
- PyTorch `1.13.0` (recommended) (compatible with: `1.11.x` - `1.13.x`)
- TensorFlow `2.8.0` (recommended) (compatible with: `2.3.x` - `2.8.x`)

### Install with PyPI (stable)

```bash
pip install netspresso
```

### Install with GitHub

To install with editable mode,

```bash
git clone https://github.com/nota-netspresso/pynetspresso.git
cd pynetspresso
pip install -e .
```

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
  --name netspresso-dev netspresso:v$(cat netspresso/VERSION)
```
