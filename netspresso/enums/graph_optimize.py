from netspresso.enums.base import StrEnum


class GraphOptimizePatternHandler(StrEnum):
    PatternHandlerFoldingGeLU = "PatternHandlerFoldingGeLU",
    PatternHandlerFoldingSilu = "PatternHandlerFoldingSilu",
    PatternHandlerFoldingRMSNorm = "PatternHandlerFoldingRMSNorm",
    PatternHandlerFoldingLayerNorm = "PatternHandlerFoldingLayerNorm",
    PatternHandlerChangeAxisOfSoftmax = "PatternHandlerChangeAxisOfSoftmax",
    PatternHandlerReplaceNegToConv = "PatternHandlerReplaceNegToConv",
    PatternHandlerFuseBNToConv = "PatternHandlerFuseBNToConv",
    PatternHandlerFuseContinuousConcat = "PatternHandlerFuseContinuousConcat",
    PatternHandlerFuseMathIntoConv = "PatternHandlerFuseMathIntoConv",
    PatternHandlerReplaceMatmulToConv = "PatternHandlerReplaceMatmulToConv",
    PatternHandlerFuseMultiReshapeTranspose = "PatternHandlerFuseMultiReshapeTranspose",
    PatternHandlerFuseMultiBranchReshapeTranspose = "PatternHandlerFuseMultiBranchReshapeTranspose",
    PatternHandlerRemoveUselessSlice = "PatternHandlerRemoveUselessSlice"

    @staticmethod
    def get_all():
        return [
            graph_optimize_pattern_handler.value
            for graph_optimize_pattern_handler in GraphOptimizePatternHandler
        ]
