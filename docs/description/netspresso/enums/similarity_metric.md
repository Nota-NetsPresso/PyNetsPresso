# SimilarityMetric

::: netspresso.enums.SimilarityMetric
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Similarity Metrics

| Name | Value | Description                    | Type       |
|------|-------|--------------------------------|------------|
| SNR  | SNR   | Signal-to-Noise Ratio metric  | Quality    |

## Example

```python
from netspresso.enums import SimilarityMetric

# Select a similarity metric
metric = SimilarityMetric.SNR

# Available metrics
print(f"SNR: {SimilarityMetric.SNR}")
```

