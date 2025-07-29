# Simulator

## Description

::: netspresso.simulator.simulator.Simulator
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Examples

### Simulating Original and Quantized Models

```python
from netspresso import NetsPresso
from pathlib import Path

netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

# Initialize simulator
simulator = netspresso.simulator()

# Simulate model performance comparison
simulate_metadata = simulator.simulate_model(
    base_model_path="./examples/sample_models/yolox-s-detection.onnx",
    target_model_path="./outputs/quantized/quantized_model.onnx",
    output_dir="./outputs/simulation"
)
```
