from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.project import ProjectIsDeletedException, ProjectNotFoundException
from netspresso.utils.db.models.project import Project
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


class ProjectRepository(BaseRepository[Project]):
    def __is_available(self, project: Optional[Project]) -> Project:
        if project is None:
            raise ProjectNotFoundException()

        if project.is_deleted:
            raise ProjectIsDeletedException(project_id=project.project_id)

        return project

    def get_by_project_id(self, db: Session, project_id: str) -> Optional[Project]:
        conditions = [self.model.project_id == project_id]
        project = self.find_first(
            db=db,
            conditions=conditions,
        )

        project = self.__is_available(project=project)

        return project

    def get_all_by_user_id(
        self,
        db: Session,
        user_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[Project]:
        conditions = [self.model.user_id == user_id]
        projects = self.find_all(
            db=db,
            conditions=conditions,
            start=start,
            size=size,
            order=order,
            time_sort=time_sort,
        )

        return projects

    def is_project_name_duplicated(self, db: Session, project_name: str, user_id: str) -> bool:
        """
        Check if a project with the same name already exists for the given API key.

        Args:
            db (Session): Database session.
            project_name (str): The name of the project to check.
            user_id (str): The ID of the user to filter the user's projects.

        Returns:
            bool: True if the project name exists, False otherwise.
        """
        conditions = [self.model.project_name == project_name, self.model.user_id == user_id]
        project = self.find_first(
            db=db,
            conditions=conditions,
        )

        return project is not None

    def count_by_user_id(self, db: Session, user_id: str) -> int:
        count_field = self.model.user_id
        conditions = [self.model.user_id == user_id]

        count = self.count_by_field(db=db, count_field=count_field, conditions=conditions)

        return count


project_repository = ProjectRepository(Project)
