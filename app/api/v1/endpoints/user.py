from fastapi import APIRouter, Depends

from app.api.deps import api_key_header
from app.api.v1.schemas.user import UserResponse
from app.services.user import user_service

router = APIRouter()


@router.get("/me", response_model=UserResponse)
def get_user(*, api_key: str = Depends(api_key_header)) -> UserResponse:
    user = user_service.get_user_info(api_key=api_key)

    return UserResponse(data=user)
