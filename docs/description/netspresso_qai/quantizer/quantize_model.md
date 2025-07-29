# Quantize Model

::: netspresso.np_qai.quantizer.NPQAIQuantizer.quantize_model
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
from netspresso.np_qai.options import QuantizePrecision

QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)

quantizer = np_qai.quantizer()

INPUT_MODEL_PATH = "YOUR_INPUT_MODEL_PATH"
OUTPUT_DIR = "YOUR_OUTPUT_DIR"
JOB_NAME = "YOUR_JOB_NAME"
DEVICE_NAME = "YOUR_DEVICE_NAME"

CALIBRATION_DATA = {"images": inputs_array}

quantized_result = quantizer.quantize_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    calibration_data=CALIBRATION_DATA,
    weights_dtype=QuantizePrecision.INT8,
    activations_dtype=QuantizePrecision.INT8,
    job_name=JOB_NAME,
)
print("Quantization task started")

# Monitor task status
while True:
    status = quantizer.get_quantize_task_status(quantized_result.quantize_info.quantize_task_uuid)
    if status.finished:
        quantized_result = quantizer.update_quantize_task(quantized_result)
        print("Quantization task completed")
        break
    else:
        print("Quantization task is still running")
```

## Create Calibration Datasets

Using good, representative input samples for calibration helps improve performance on target hardware and retains model accuracy

```python
import cv2
from glob import glob

import numpy as np
from netspresso.inferencer.preprocessors.base import Preprocessor

IMG_SIZE = 640
preprocess_list = [
    {
        "name": "resize",
        "size": IMG_SIZE,
        "interpolation": "bilinear",
        "max_size": None,
        "resize_criteria": "long",
    },
    {
        "name": "pad",
        "size": IMG_SIZE,
        "fill": 114,
    }
]

preprocessor = Preprocessor(preprocess_list)

DATASET_PATH = "YOUR_DATASET_PATH"
NUM_DATASET = 100
image_paths = glob(f"{DATASET_PATH}/*.jpg")[:NUM_DATASET]

inputs_array = []

for image_path in image_paths:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessor(img)
    img = np.transpose(img, (0, 3, 1, 2))
    inputs_array.append(img)

>> np.array(inputs_array).shape
(100, 1, 3, 512, 512)
``` 