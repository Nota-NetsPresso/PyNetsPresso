<div align=right>
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FNota-NetsPresso%2Fnetspresso-python&count_bg=%2323E7E7E7&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>

# NetsPresso Python Package

<div align="center">
    <img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/banner/pynp_main.png"/>
</div>
</br>

<div align="center">
    <a href="https://github.com/Nota-NetsPresso">
        <img alt="github" src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/github.png" width="3%">
    </a>
    <img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/logo-transparent.png" width="3%"/>
    <a href="https://www.facebook.com/NotaAI">
        <img alt="facebook" src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/facebook.png" width="3%">
    </a>
    <img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/logo-transparent.png" width="3%"/>
    <a href="https://twitter.com/nota_ai">
        <img alt="twitter" src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/twitter.png" width="3%">
    </a>
    <img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/logo-transparent.png" width="3%"/>
    <a href="https://www.youtube.com/channel/UCeewYFAqb2EqwEXZCfH9DVQ">
        <img alt="youtube" src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/youtube.png" width="3%">
    </a>
    <img src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/logo-transparent.png" width="3%"/>
    <a href="https://www.linkedin.com/company/nota-incorporated">
        <img alt="youtube" src="https://netspresso-docs-imgs.s3.ap-northeast-2.amazonaws.com/imgs/common/linkedin.png" width="3%">
    </a>
</div>
</br>

<div align="center">
    <p align="center">
        <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue" />
        <a href="https://www.tensorflow.org/install/pip" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-2.3.x ~ 2.5.x.-FF6F00?style=flat&logo=tensorflow&logoColor=#FF6F00&link=https://www.tensorflow.org/install/pip"/></a>
        <a href="https://pytorch.org/" target="_blank"><img src="https://img.shields.io/badge/PyTorch-1.11.x ~ 1.13.x.-EE4C2C?style=flat&logo=pytorch&logoColor=#EE4C2C"/></a>
    </p>
</div>
</br>

Use **PyNetsPresso** for a seamless model optimization process. 
PyNetsPresso resolves AI-related constraints in business use cases and enables cost-efficiency and enhanced performance by removing the requirement for high-spec servers and network connectivity and preventing high latency and personal data breaches.

The **NetsPresso Python package** is a python interface with the NetsPresso web application and REST API.

Easily compress various models with our resources. Please browse the [Docs] for details, and join our [Discussion Forum] for providing feedback or sharing your use cases.

To get started with the NetsPresso Python package, you will need to sign up either at [NetsPresso] or [PyNetsPresso].</a>


## Installation

There are two ways you can install the NetsPresso Python Package: using pip or manually through our project GitHub repository.

To install this package, please use Python 3.8 or higher.

From PyPI (Recommended)
```bash
pip install netspresso
```

From Github
```bash
git clone https://github.com/Nota-NetsPresso/netspresso-python.git
pip install -e .
```

## Quick Start

### Login

To use the NetsPresso Python package, please enter the email and password registered in NetsPresso.

```python
from netspresso.compressor import ModelCompressor

compressor = ModelCompressor(email="YOUR_EMAIL", password="YOUR_PASSWORD")
```

### Upload Model

To upload your trained model, simply enter the required information. 

When a model is successfully uploaded, a unique 'model.model_id' is generated to allow repeated use of the uploaded model.

```python
from netspresso.compressor import Task, Framework

model = compressor.upload_model(
    model_name="YOUR_MODEL_NAME",
    task=Task.IMAGE_CLASSIFICATION,
    framework=Framework.TENSORFLOW_KERAS,
    file_path="YOUR_MODEL_PATH", # ex) ./model.h5
    input_shapes="YOUR_MODEL_INPUT_SHAPES",  # ex) [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
)
```

### Automatic Compression

Automatically compress the model by setting the compression ratio for the model.

Enter the ID of the uploaded model, the name and storage path of the compressed model, and the compression ratio.

```python
compressed_model = compressor.automatic_compression(
    model_id=model.model_id,
    model_name="YOUR_COMPRESSED_MODEL_NAME",
    output_path="OUTPUT_PATH",  # ex) ./compressed_model.h5
    compression_ratio=0.5,
)
```

## Contact

Join our [Discussion Forum] for providing feedback or sharing your use cases, and if you want to talk more with Nota, please contact us [here].</br>
Or you can also do it via email([contact@nota.ai]) or phone(+82 2-555-8659)!


[Docs]: https://nota-netspresso.github.io/PyNP-python-package-docs/build/html/index.html
[Discussion Forum]: https://github.com/orgs/Nota-NetsPresso/discussions
[NetsPresso]: https://netspresso.ai?utm_source=git_comp&utm_medium=text_np&utm_campaign=py_launch
[PyNetsPresso]: https://py.netspresso.ai/?utm_source=git_comp&utm_medium=text_py&utm_campaign=py_launch
[here]: https://www.nota.ai/contact-us
[contact@nota.ai]: mailto:contact@nota.ai
