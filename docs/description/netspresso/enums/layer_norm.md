# LayerNorm

::: netspresso.enums.LayerNorm
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Layer Normalization Methods

| Name            | Value           | Description                      |
|-----------------|-----------------|----------------------------------|
| NONE            | none            | No layer normalization           |
| STANDARD_SCORE  | standard_score  | Standard score normalization     |
| TSS_NORM        | tss_norm        | TSS normalization method         |
| LINEAR_SCALING  | linear_scaling  | Linear scaling normalization     |
| SOFTMAX_NORM    | softmax_norm    | Softmax normalization method     |

## Example

```python
from netspresso.enums import LayerNorm

# Select a layer normalization method
layer_norm = LayerNorm.STANDARD_SCORE

# Available methods
print(f"None: {LayerNorm.NONE}")
print(f"Standard Score: {LayerNorm.STANDARD_SCORE}")
print(f"TSS Norm: {LayerNorm.TSS_NORM}")
print(f"Linear Scaling: {LayerNorm.LINEAR_SCALING}")
print(f"Softmax Norm: {LayerNorm.SOFTMAX_NORM}")
```

