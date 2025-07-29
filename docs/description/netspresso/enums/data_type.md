# DataType

::: netspresso.enums.DataType
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Data Types

| Name    | Value  | Description                   | Bits |
|---------|--------|-------------------------------|------|
| FP32    | FP32   | 32-bit floating point        | 32   |
| FP16    | FP16   | 16-bit floating point        | 16   |
| INT8    | INT8   | 8-bit signed integer         | 8    |
| NONE    | ""     | No data type specified       | -    |

## Example

```python
from netspresso.enums import DataType

# Select a data type
data_type = DataType.FP32

# Available data types
print(f"FP32: {DataType.FP32}")
print(f"FP16: {DataType.FP16}")
print(f"INT8: {DataType.INT8}")
print(f"None: {DataType.NONE}")
```

