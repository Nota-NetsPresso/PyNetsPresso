from enum import Enum
from typing import Literal


class Module(str, Enum):
    CONVERT = "CONVERT"
    BENCHMARK = "BENCHMARK"

    @classmethod
    def create_literal(cls):
        return Literal["CONVERT", "BENCHMARK"]
