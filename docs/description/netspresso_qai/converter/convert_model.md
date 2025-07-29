# Convert Model

::: netspresso.np_qai.converter.NPQAIConverter.convert_model
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Example

```python
from netspresso import NPQAI
from netspresso.np_qai import Device
from netspresso.np_qai.options import CompileOptions, Runtime, ComputeUnit, QuantizeFullType

QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)

converter = np_qai.converter()

convert_options = CompileOptions(
    target_runtime=Runtime.TFLITE,
    compute_unit=[ComputeUnit.NPU],
    quantize_full_type=QuantizeFullType.INT8,
    quantize_io=True,
    quantize_io_type=QuantizeFullType.INT8,
)

IMG_SIZE = 640
INPUT_MODEL_PATH = "YOUR_INPUT_MODEL_PATH"
OUTPUT_DIR = "YOUR_OUTPUT_DIR"
JOB_NAME = "YOUR_JOB_NAME"
DEVICE_NAME = "QCS6490 (Proxy)"
converted_result = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_device_name=Device(DEVICE_NAME),
    options=convert_options,
    input_shapes=dict(image=(1, 3, IMG_SIZE, IMG_SIZE)),
    job_name=JOB_NAME,
)

print("Conversion task started")

# Monitor task status
while True:
    status = converter.get_convert_task_status(converted_result.convert_task_info.convert_task_uuid)
    if status.finished:
        converted_result = converter.update_convert_task(converted_result)
        print("Conversion task completed")
        break
    else:
        print("Conversion task is still running")
``` 