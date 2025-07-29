# Graph Optimize

Graph optimization enums for configuring pattern handlers in NetsPresso's graph optimizer.

## GraphOptimizePatternHandler

::: netspresso.enums.graph_optimize.GraphOptimizePatternHandler
    handler: python
    options:
      heading_level: 3
      show_root_heading: true
      show_source: false
      show_symbol_type_toc: true

## Available Pattern Handlers

| Category | Pattern Name | Status |
|----------|--------------|--------|
| **Fuse** | `PatternHandlerFoldingGeLU` | ✓ |
| **Fuse** | `PatternHandlerFoldingSilu` | ✓ |
| **Fuse** | `PatternHandlerFoldingRMSNorm` | ✓ |
| **Fuse** | `PatternHandlerFoldingLayerNorm` | ✓ |
| **Optimize** | `PatternHandlerChangeAxisOfSoftmax` | ✓ |
| **Replace** | `PatternHandlerReplaceNegToConv` | ✓ |
| **Fuse** | `PatternHandlerFuseBNToConv` | ✓ |
| **Fuse** | `PatternHandlerFuseContinuousConcat` | ✓ |
| **Fuse** | `PatternHandlerFuseMathIntoConv` | ✓ |
| **Replace** | `PatternHandlerReplaceMatmulToConv` | ✓ |
| **Fuse** | `PatternHandlerFuseMultiReshapeTranspose` | ✓ |
| **Fuse** | `PatternHandlerFuseMultiBranchReshapeTranspose` | ✓ |
| **Remove** | `PatternHandlerRemoveUselessSlice` | ✓ |

## Usage Examples

### Get All Available Pattern Handlers

```python
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler

# Get all available pattern handlers
all_patterns = GraphOptimizePatternHandler.get_all()
print(f"Available patterns: {len(all_patterns)}")
```
