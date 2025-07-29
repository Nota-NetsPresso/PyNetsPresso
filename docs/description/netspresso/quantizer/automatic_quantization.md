# Automatic Quantization

## Description

::: netspresso.quantizer.quantizer.Quantizer
    handler: python
    options:
      members:
        - automatic_quantization
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Examples

```python
from netspresso import NetsPresso
from netspresso.enums import QuantizationPrecision


# Login with API key (recommended)
# Get your API token from: https://account.netspresso.ai/api-token
netspresso = NetsPresso(api_key="YOUR_API_KEY")

# Note: Email/password login will be deprecated soon
# netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

quantizer = netspresso.quantizer()
quantization_result = quantizer.automatic_quantization(
    input_model_path="./examples/sample_models/test.onnx",
    output_dir="./outputs/quantized/automatic_quantization",
    dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
    weight_precision=QuantizationPrecision.INT8,
    activation_precision=QuantizationPrecision.INT8,
    threshold=0,
)
```
