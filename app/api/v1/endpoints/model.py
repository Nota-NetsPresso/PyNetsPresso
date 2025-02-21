from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.model import ModelDetailResponse, ModelsResponse
from app.services.model import model_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get("", response_model=ModelsResponse)
def get_models(
    *,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ModelsResponse:
    models = model_service.get_models(db=db, api_key=api_key)

    return ModelsResponse(data=models, total_count=len(models))


@router.get("/{model_id}", response_model=ModelDetailResponse)
def get_model(
    *,
    model_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ModelDetailResponse:
    model = model_service.get_model(db=db, model_id=model_id, api_key=api_key)

    return ModelDetailResponse(data=model)


@router.delete("/{model_id}", response_model=ModelDetailResponse)
def delete_model(
    *,
    model_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ModelDetailResponse:
    model = model_service.delete_model(db=db, model_id=model_id, api_key=api_key)

    return ModelDetailResponse(data=model)
