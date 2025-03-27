import sys
from enum import Enum

# Check if running on Python 3.11 or later
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Define a fallback StrEnum for Python 3.10
    class StrEnum(str, Enum):
        pass
