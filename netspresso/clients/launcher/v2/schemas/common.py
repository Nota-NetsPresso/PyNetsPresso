from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field


class Order(str, Enum):
    """ """

    DESC = "desc"
    ASC = "asc"


@dataclass(init=False)
class AuthorizationHeader:
    Authorization: str

    def __init__(self, access_token):
        self.Authorization = f"Bearer {access_token}"


@dataclass(init=False)
class UploadFile:
    files: List

    def __init__(self, file_name, file_content):
        self.files = [("file", (file_name, file_content))]


@dataclass
class RequestPagination:
    """ """

    start: int = 0
    size: int = 10
    order: Order = Order.DESC.value
    paging: bool = True


@dataclass
class ResponseItem:
    """ """

    data: Optional[object] = field(default_factory=dict)


@dataclass
class ResponseItems:
    """ """

    data: List[Optional[object]] = field(default_factory=list)


@dataclass
class ResponsePaginationItems:
    """ """

    result_count: int
    total_count: int
    data: List[Optional[object]] = field(default_factory=list)
