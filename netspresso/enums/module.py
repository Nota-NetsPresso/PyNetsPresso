from typing import Literal

from netspresso.enums.base import StrEnum


class Module(StrEnum):
    CONVERT = "CONVERT"
    BENCHMARK = "BENCHMARK"

    @classmethod
    def create_literal(cls):
        return Literal["CONVERT", "BENCHMARK"]
