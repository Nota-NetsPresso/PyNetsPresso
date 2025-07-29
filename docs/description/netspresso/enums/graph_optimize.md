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

| Category | Pattern Name | Status | Description |
|----------|--------------|--------|-------------|
| **Fuse** | `PatternHandlerFuseBNToConv` | ✓ | Fuse batch normalization to convolution layers |
| **Fuse** | `PatternHandlerFuseMathIntoConv` | ✓ | Fuse math operators (add, sub, mul, div) to convolution recursively |
| **Fuse** | `PatternHandlerFuseMultiReshapeTranspose` | ✓ | Fuse multiple reshapes and transposes into one |
| **Fuse** | `PatternHandlerFuseMultiBranchReshapeTranspose` | To do | Fuse multiple branched reshapes and transposes into one |
| **Fuse** | `PatternHandlerFuseContinuousConcat` | ✓ | Fuse continuous concatenation operations |
| **Fuse** | `PatternHandlerFoldingGeLU` | ✓ | Optimize GeLU activation functions by folding |
| **Fuse** | `PatternHandlerFoldingSilu` | ✓ | Optimize SiLU activation functions by folding |
| **Fuse** | `PatternHandlerFoldingRMSNorm` | ✓ | Optimize RMS normalization layers by folding |
| **Fuse** | `PatternHandlerFoldingLayerNorm` | ✓ | Optimize layer normalization operations by folding |
| **Remove** | `PatternHandlerRemoveUselessSlice` | ✓ | Remove meaningless tensor slice operators |
| **Replace** | `PatternHandlerReplaceNegToConv` | ✓ | Replace negation by depthwise convolution |
| **Replace** | `PatternHandlerReplaceMatmulToConv` | ✓ | Replace matrix multiplication with convolution operations |
| **Optimize** | `PatternHandlerChangeAxisOfSoftmax` | ✓ | Optimize softmax operations by changing axis configurations |
