# Framework

::: netspresso.enums.Framework
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Frameworks

| Name             | Value            | Description                      | File Extension |
|------------------|------------------|----------------------------------|----------------|
| TENSORFLOW_KERAS | tensorflow_keras | TensorFlow-Keras                 | .h5, .zip      |
| TENSORFLOW       | saved_model      | TensorFlow SavedModel            | .zip           |
| PYTORCH          | pytorch          | PyTorch                          | .pt            |
| ONNX             | onnx             | ONNX                             | .onnx          |
| TENSORRT         | tensorrt         | TensorRT                         | .engine        |
| OPENVINO         | openvino         | OpenVINO                         | .xml           |
| TENSORFLOW_LITE  | tensorflow_lite  | TensorFlow Lite                  | .tflite        |
| DRPAI            | drpai            | DRP-AI                           | .onnx          |

## Framework Support

!!! note
    - **Compressor Supported**: TENSORFLOW_KERAS, PYTORCH, ONNX
    - **Launcher Supported**: ONNX, TENSORRT, OPENVINO, TENSORFLOW_LITE, DRPAI, KERAS, SAVED_MODEL
    - **ONNX (.onnx)**: Supported version PyTorch >= 1.11.x, ONNX >= 1.10.x
    - **PyTorch (.pt)**: Supported version PyTorch >= 1.11.x
    - **TensorFlow-Keras (.h5, .zip)**: Supported version TensorFlow 2.3.x ~ 2.8.x

## Example

```python
from netspresso.enums import Framework

# Select a framework
framework = Framework.PYTORCH

# Available frameworks
print(f"TensorFlow-Keras: {Framework.TENSORFLOW_KERAS}")
print(f"PyTorch: {Framework.PYTORCH}")
print(f"ONNX: {Framework.ONNX}")
print(f"TensorRT: {Framework.TENSORRT}")
```
