# QuantizationMode

::: netspresso.enums.QuantizationMode
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Quantization Modes

| Name                              | Value                             | Description                                     |
|-----------------------------------|-----------------------------------|-------------------------------------------------|
| AUTOMATIC_QUANTIZATION           | automatic_quantization           | Automatic quantization mode                    |
| UNIFORM_PRECISION_QUANTIZATION   | uniform_precision_quantization   | Uniform precision quantization mode            |
| CUSTOM_PRECISION_QUANTIZATION    | custom_precision_quantization    | Custom precision quantization mode             |
| ADVANCED_QUANTIZATION            | advanced_quantization            | Advanced quantization mode                     |
| RECOMMEND_QUANTIZATION           | recommend_quantization           | Recommend quantization mode                    |

## Example

```python
from netspresso.enums import QuantizationMode

# Select a quantization mode
mode = QuantizationMode.AUTOMATIC_QUANTIZATION

# Available modes
print(f"Automatic: {QuantizationMode.AUTOMATIC_QUANTIZATION}")
print(f"Uniform Precision: {QuantizationMode.UNIFORM_PRECISION_QUANTIZATION}")
print(f"Custom Precision: {QuantizationMode.CUSTOM_PRECISION_QUANTIZATION}")
print(f"Advanced: {QuantizationMode.ADVANCED_QUANTIZATION}")
print(f"Recommend: {QuantizationMode.RECOMMEND_QUANTIZATION}")
```

