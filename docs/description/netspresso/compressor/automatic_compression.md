# Automatic Compression

## Description

::: netspresso.compressor.v2.compressor.CompressorV2.automatic_compression
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

### Details of Parameters

#### Compression Ratio

!!! note
    - As the compression ratio increases, you can get more lighter and faster compressed models, but with greater lost of accuracy. 
    - Therefore, it is necessary to find an appropriate ratio for your requirements. It might require a few trials and errors.
    - The range of available values is as follows.

    $$0 < \text{ratio} < 1$$

## Example

```python
from netspresso import NetsPresso


netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

compressor = netspresso.compressor_v2()
compressed_model = compressor.automatic_compression(
    input_shapes=[{"batch": 1, "channel": 3, "dimension": [224, 224]}],
    input_model_path="./examples/sample_models/graphmodule.pt",
    output_dir="./outputs/compressed/pytorch_automatic_compression_1",
    compression_ratio=0.5,
)
```
