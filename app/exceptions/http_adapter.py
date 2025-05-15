from fastapi import status

from app.exceptions.base import ExceptionBase
from app.exceptions.schema import AdditionalData, Origin
from netspresso.exceptions.common import AdditionalData as PyNPAdditionalData
from netspresso.exceptions.common import PyNPException
from netspresso.exceptions.status import STATUS_MAP


class HTTPExceptionAdapter(ExceptionBase):
    """
    PyNPException을 HTTPException으로 변환하는 어댑터 클래스
    """

    @classmethod
    def from_pynp_exception(cls, exc: PyNPException):
        """
        PyNPException 객체를 HTTPExceptionAdapter 객체로 변환합니다.

        Args:
            exc: 변환할 PyNPException 객체

        Returns:
            HTTPExceptionAdapter: 변환된 HTTPException 객체
        """
        error_code = exc.detail.get("error_code", "")
        status_code = STATUS_MAP.get(error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        name = exc.detail.get("name", "")
        message = exc.detail.get("message", "")

        # PyNPAdditionalData에서 AdditionalData로 변환
        pynp_data = exc.detail.get("data", {})
        origin_str = pynp_data.get("origin", "")

        try:
            origin = Origin(origin_str) if origin_str in [e.value for e in Origin] else None
        except ValueError:
            origin = None

        data = AdditionalData(
            origin=origin,
            error_log=pynp_data.get("error_log", "")
        )

        return cls(
            data=data,
            error_code=error_code,
            status_code=status_code,
            name=name,
            message=message
        )
