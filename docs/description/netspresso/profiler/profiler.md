# Profiler

## Description

::: netspresso.profiler.profiler.Profiler
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Examples

### Profiling converted model on Jetson AGX Orin

```python
from netspresso import NetsPresso
from netspresso.enums import DeviceName, SoftwareVersion


netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

profiler = netspresso.profiler()
profiling_task = profiler.profile_model(
   input_model_path="./outputs/converted/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1.trt",
   target_device_name=DeviceName.JETSON_AGX_ORIN,
   target_software_version=SoftwareVersion.JETPACK_5_0_1,
)
```
