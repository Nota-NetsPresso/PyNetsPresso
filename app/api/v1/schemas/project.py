from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from netspresso.enums import Status
from netspresso.exceptions.project import ProjectNameTooLongException


class ProjectCreate(BaseModel):
    project_name: str = Field(..., description="The name of the project to be created.")

    @field_validator("project_name")
    def validate_length_of_project_name(cls, project_name: str) -> str:
        if len(project_name) > 30:
            raise ProjectNameTooLongException(max_length=30, actual_length=len(project_name))
        return project_name


class ProjectDuplicationStatus(BaseModel):
    is_duplicated: bool = Field(..., description="Indicates if the project name is duplicated.")


class ProjectPayload(ProjectCreate):
    model_config = ConfigDict(from_attributes=True)

    project_id: str = Field(..., description="The unique identifier for the project.")
    model_ids: List[str] = Field(default_factory=list, description="The list of models associated with the project.")
    user_id: str = Field(..., description="The unique identifier for the user associated with the project.")
    project_abs_path: str = Field(..., description="The absolute path of the project.")
    created_at: datetime = Field(..., description="The timestamp when the project was created.")
    updated_at: datetime = Field(..., description="The timestamp when the project was last updated.")


class ProjectDuplicationCheckResponse(ResponseItem):
    data: ProjectDuplicationStatus


class ProjectResponse(ResponseItem):
    data: ProjectPayload


class ProjectsResponse(ResponsePaginationItems):
    data: List[ProjectPayload]
