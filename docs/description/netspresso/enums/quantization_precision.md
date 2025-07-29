# QuantizationPrecision

::: netspresso.enums.QuantizationPrecision
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Quantization Precisions

| Name    | Value   | Description                | Bits |
|---------|---------|----------------------------|------|
| INT8    | int8    | 8-bit integer precision    | 8    |
| FLOAT16 | float16 | 16-bit floating precision  | 16   |
| FLOAT32 | float32 | 32-bit floating precision  | 32   |

## Example

```python
from netspresso.enums import QuantizationPrecision

# Select a quantization precision
precision = QuantizationPrecision.INT8

# Available precisions
print(f"INT8: {QuantizationPrecision.INT8}")
print(f"FLOAT16: {QuantizationPrecision.FLOAT16}")
print(f"FLOAT32: {QuantizationPrecision.FLOAT32}")
```

