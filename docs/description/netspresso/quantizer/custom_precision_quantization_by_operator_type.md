# Quantizer(Custom Precision Quantization by Operator Type)

## Description

::: netspresso.quantizer.quantizer.Quantizer
    handler: python
    options:
      members:
        - custom_precision_quantization_by_operator_type
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Examples

```python
from netspresso import NetsPresso
from netspresso.enums import QuantizationPrecision


netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

quantizer = netspresso.quantizer()

recommendation_metadata = quantizer.get_recommendation_precision(
    input_model_path="./examples/sample_models/test.onnx",
    output_dir="./outputs/quantized/automatic_quantization",
    dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
    weight_precision=QuantizationPrecision.INT8,
    activation_precision=QuantizationPrecision.INT8,
    threshold=0,
)
recommendation_precisions = quantizer.load_recommendation_precision_result(recommendation_metadata.recommendation_result_path)

quantization_result = quantizer.custom_precision_quantization_by_operator_type(
    input_model_path="./examples/sample_models/test.onnx",
    output_dir="./outputs/quantized/custom_precision_quantization_by_operator_type",
    dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
    precision_by_operator_type=recommendation_precisions.operators,
)
```
