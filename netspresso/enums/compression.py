from typing import Literal

from netspresso.enums.base import StrEnum


class CompressionMethod(StrEnum):
    PR_L2 = "PR_L2"
    PR_GM = "PR_GM"
    PR_NN = "PR_NN"
    PR_SNP = "PR_SNP"
    PR_ID = "PR_ID"
    FD_TK = "FD_TK"
    FD_CP = "FD_CP"
    FD_SVD = "FD_SVD"

    @classmethod
    def create_literal(cls):
        return Literal["PR_L2", "PR_GM", "PR_NN", "PR_SNP", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"]


class RecommendationMethod(StrEnum):
    SLAMP = "slamp"
    VBMF = "vbmf"

    @classmethod
    def create_literal(cls):
        return Literal["slamp", "vbmf"]


class Policy(StrEnum):
    SUM = "sum"
    AVERAGE = "average"

    @classmethod
    def create_literal(cls):
        return Literal["sum", "average"]


class GroupPolicy(StrEnum):
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    NONE = "none"

    @classmethod
    def create_literal(cls):
        return Literal["sum", "average", "count", "none"]


class LayerNorm(StrEnum):
    NONE = "none"
    STANDARD_SCORE = "standard_score"
    TSS_NORM = "tss_norm"
    LINEAR_SCALING = "linear_scaling"
    SOFTMAX_NORM = "softmax_norm"

    @classmethod
    def create_literal(cls):
        return Literal["none", "standard_score", "tss_norm", "linear_scaling", "softmax_norm"]


class StepOp(StrEnum):
    ROUND_UP = "round_up"
    ROUND_DOWN = "round_down"
    ROUND = "round"
    NONE = "none"

    @classmethod
    def create_literal(cls):
        return Literal["round_up", "round_down", "round", "none"]


compression_literal = CompressionMethod.create_literal()
recommendation_literal = RecommendationMethod.create_literal()
policy_literal = Policy.create_literal()
grouppolicy_literal = GroupPolicy.create_literal()
layernorm_literal = LayerNorm.create_literal()
stepop_literal = StepOp.create_literal()
