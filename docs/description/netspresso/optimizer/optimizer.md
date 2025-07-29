# Optimizer

## Description

::: netspresso.graph_optimizer.graph_optimizer.GraphOptimizer
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Pattern Handlers

The following pattern handlers are available for graph optimization:

::: netspresso.enums.graph_optimize.GraphOptimizePatternHandler
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

### Pattern Handler Types

- **PatternHandlerFoldingGeLU**
- **PatternHandlerFoldingSilu**
- **PatternHandlerFoldingRMSNorm**
- **PatternHandlerFoldingLayerNorm**
- **PatternHandlerChangeAxisOfSoftmax**
- **PatternHandlerReplaceNegToConv**
- **PatternHandlerFuseBNToConv**
- **PatternHandlerFuseContinuousConcat**
- **PatternHandlerFuseMathIntoConv**
- **PatternHandlerReplaceMatmulToConv**
- **PatternHandlerFuseMultiReshapeTranspose**
- **PatternHandlerFuseMultiBranchReshapeTranspose**
- **PatternHandlerRemoveUselessSlice**

## Examples

### Basic Graph Optimization

```python
from netspresso import NetsPresso
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler

netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

# Initialize graph optimizer
optimizer = netspresso.graph_optimizer()

# Optimize model with specific pattern handlers
optimized_model = optimizer.optimize(
    input_model_path="./examples/sample_models/test.onnx",
    output_dir="./outputs/optimized/",
    pattern_handlers=[
        GraphOptimizePatternHandler.PatternHandlerFuseBNToConv,
        GraphOptimizePatternHandler.PatternHandlerFoldingLayerNorm,
        GraphOptimizePatternHandler.PatternHandlerRemoveUselessSlice
    ]
)
```

### Using All Available Pattern Handlers

```python
from netspresso import NetsPresso
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler

netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

# Initialize graph optimizer
optimizer = netspresso.graph_optimizer()

# Get all available pattern handlers
all_handlers = GraphOptimizePatternHandler.get_all()

# Optimize model with all pattern handlers
optimized_model = optimizer.optimize(
    input_model_path="./examples/sample_models/test.onnx",
    output_dir="./outputs/optimized/",
    pattern_handlers=all_handlers
)
```
