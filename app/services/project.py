from typing import List, Optional

from sqlalchemy.orm import Session

from app.api.v1.schemas.project import ProjectPayload
from app.services.user import user_service
from netspresso.utils.db.models.project import Project
from netspresso.utils.db.repositories.base import Order
from netspresso.utils.db.repositories.project import project_repository


class ProjectService:
    def create_project(self, db: Session, project_name: str, api_key: str) -> Project:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        project = netspresso.create_project(project_name=project_name)
        project = ProjectPayload.model_validate(project)

        return project

    def check_project_duplication(self, db: Session, project_name: str, api_key: str) -> bool:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        is_duplicated = project_repository.is_project_name_duplicated(
            db=db, project_name=project_name, user_id=netspresso.user_info.user_id
        )

        return is_duplicated

    def get_projects(
        self, *, db: Session, start: Optional[int], size: Optional[int], order: Order, api_key: str
    ) -> List[Project]:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        projects = project_repository.get_all_by_user_id(
            db=db, user_id=netspresso.user_info.user_id, start=start, size=size, order=order
        )
        projects = [ProjectPayload.model_validate(project) for project in projects]

        return projects

    def count_project_by_user_id(self, *, db: Session, api_key: str) -> int:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        return project_repository.count_by_user_id(db=db, user_id=netspresso.user_info.user_id)

    def get_project(self, *, db: Session, project_id: str, api_key: str) -> Project:
        # netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        project = project_repository.get_by_project_id(db=db, project_id=project_id)
        project = ProjectPayload.model_validate(project)

        return project


project_service = ProjectService()
