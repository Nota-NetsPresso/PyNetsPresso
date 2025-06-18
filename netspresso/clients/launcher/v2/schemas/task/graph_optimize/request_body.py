from dataclasses import dataclass, field
from typing import List

from netspresso.enums.graph_optimize import GraphOptimizePatternHandler


@dataclass
class RequestCreateGraphOptimizeTask:
    ai_model_id: str
    pattern_handlers: List[GraphOptimizePatternHandler] = field(
        default_factory=GraphOptimizePatternHandler.get_all
    )
