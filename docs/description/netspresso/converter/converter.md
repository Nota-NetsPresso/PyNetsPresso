# Trainer

## Description

::: netspresso.converter.v2.converter.ConverterV2
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Examples

### Converting a model to TensorRT on Jetson AGX Orin

```python
from netspresso import NetsPresso
from netspresso.enums import DeviceName, Framework, SoftwareVersion


netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

converter = netspresso.converter_v2()
conversion_task = converter.convert_model(
   input_model_path="./examples/sample_models/test.onnx",
   output_dir="./outputs/converted/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1",
   target_framework=Framework.TENSORRT,
   target_device_name=DeviceName.JETSON_AGX_ORIN,
   target_software_version=SoftwareVersion.JETPACK_5_0_1,
)
```
