# Extension

::: netspresso.enums.Extension
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available File Extensions

| Name | Value | Description                      | Framework    |
|------|-------|----------------------------------|--------------|
| H5   | h5    | Keras HDF5 model file           | TensorFlow   |
| ZIP  | zip   | TensorFlow SavedModel archive    | TensorFlow   |
| PT   | pt    | PyTorch model file              | PyTorch      |
| ONNX | onnx  | ONNX model file                 | ONNX         |

## Example

```python
from netspresso.enums import Extension

# Select an extension
extension = Extension.ONNX

# Available extensions
print(f"H5: {Extension.H5}")
print(f"ZIP: {Extension.ZIP}")
print(f"PT: {Extension.PT}")
print(f"ONNX: {Extension.ONNX}")
```

