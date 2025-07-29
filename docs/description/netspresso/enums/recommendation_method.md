# RecommendationMethod

::: netspresso.enums.RecommendationMethod
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Recommendation Methods

| Name  | Value | Description                                    | Type              |
|-------|-------|------------------------------------------------|-------------------|
| SLAMP | slamp | SLAMP recommendation method                    | Structured        |
| VBMF  | vbmf  | Variational Bayesian Matrix Factorization     | Factorization     |

## Example

```python
from netspresso.enums import RecommendationMethod

# Select a recommendation method
method = RecommendationMethod.SLAMP

# Available methods
print(f"SLAMP: {RecommendationMethod.SLAMP}")
print(f"VBMF: {RecommendationMethod.VBMF}")
```
