# Compression Method

::: netspresso.enums.CompressionMethod
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Compression Methods

| Name   | Value  | Description                      | Type          |
|--------|--------|----------------------------------|---------------|
| PR_L2  | PR_L2  | L2 Norm Pruning                  | Pruning       |
| PR_GM  | PR_GM  | GM Pruning                       | Pruning       |
| PR_NN  | PR_NN  | Nuclear Norm Pruning             | Pruning       |
| PR_SNP | PR_SNP | Structured Neuron-level Pruning  | Pruning       |
| PR_ID  | PR_ID  | Pruning By Index                 | Pruning       |
| FD_TK  | FD_TK  | Tucker Decomposition             | Factorization |
| FD_SVD | FD_SVD | Singular Value Decomposition     | Factorization |
| FD_CP  | FD_CP  | CP Decomposition                 | Factorization |

## Framework Compatibility

!!! warning
    - **Nuclear Norm (PR_NN)** is only supported in the Tensorflow-Keras framework.
    - **Structured Neuron-level (PR_SNP)** is only supported in the PyTorch and ONNX frameworks.
    - All other methods are supported across all frameworks.

## Example

```python
from netspresso.enums import CompressionMethod

# Select a compression method
compression_method = CompressionMethod.PR_L2

# Available methods
print(f"L2 Norm Pruning: {CompressionMethod.PR_L2}")
print(f"GM Pruning: {CompressionMethod.PR_GM}")
print(f"Tucker Decomposition: {CompressionMethod.FD_TK}")
print(f"CP Decomposition: {CompressionMethod.FD_CP}")
```
