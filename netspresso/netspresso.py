from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from netspresso.benchmarker import BenchmarkerV2
from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.tao import TAOTokenHandler
from netspresso.compressor import CompressorV2
from netspresso.constant.project import SUB_FOLDERS
from netspresso.converter import ConverterV2
from netspresso.enums import Task
from netspresso.exceptions.project import (
    ProjectAlreadyExistsException,
    ProjectNameTooLongException,
    ProjectSaveException,
)
from netspresso.inferencer.inferencer import CustomInferencer, NPInferencer
from netspresso.quantizer import Quantizer
from netspresso.tao import TAOTrainer
from netspresso.trainer import Trainer
from netspresso.utils.db.models.project import Project
from netspresso.utils.db.repositories.project import project_repository
from netspresso.utils.db.session import SessionLocal


class NetsPresso:
    def __init__(self, email: str, password: str, verify_ssl: bool = True) -> None:
        """Initialize NetsPresso instance and perform user authentication.

        Args:
            email (str): User's email for authentication.
            password (str): User's password for authentication.
            verify_ssl (bool): Flag to indicate whether SSL certificates should be verified. Defaults to True.
        """
        self.token_handler = TokenHandler(email=email, password=password, verify_ssl=verify_ssl)
        self.user_info = self.get_user()

    def get_user(self) -> UserResponse:
        """Get user information using the access token.

        Returns:
            UserInfo: User information.
        """
        user_info = auth_client.get_user_info(self.token_handler.tokens.access_token, self.token_handler.verify_ssl)
        return user_info

    def create_project(self, project_name: str, project_path: str = "./projects") -> Project:
        """
        Create a new project with the specified name and path.

        This method creates a project directory structure on the file system
        and saves the project information in the database. It also handles
        scenarios where the project name is too long or already exists.

        Args:
            project_name (str): The name of the project to create.
                Must not exceed 30 characters.
            project_path (str, optional): The base path where the project
                will be created. Defaults to "./projects".

        Returns:
            Project: The created project object containing information
            such as project name, user ID, and absolute path.

        Raises:
            ProjectNameTooLongException: If the `project_name` exceeds the
                maximum allowed length of 30 characters.
            ProjectAlreadyExistsException: If a project with the same name
                already exists at the specified `project_path`.
            ProjectSaveException: If an error occurs while saving the project
                to the database.
        """
        if len(project_name) > 30:
            raise ProjectNameTooLongException(max_length=30, actual_length=len(project_name))

        # Create the main project folder
        project_folder_path = Path(project_path) / project_name

        # Check if the project folder already exists
        if project_folder_path.exists():
            logger.warning(f"Project '{project_name}' already exists at {project_folder_path.resolve()}.")
            raise ProjectAlreadyExistsException(
                project_name=project_name, project_path=project_folder_path.resolve().as_posix()
            )
        else:
            project_folder_path.mkdir(parents=True, exist_ok=True)
            project_abs_path = project_folder_path.resolve()

            # Create subfolders
            for folder in SUB_FOLDERS:
                (project_folder_path / folder).mkdir(parents=True, exist_ok=True)

            logger.info(f"Project '{project_name}' created at {project_abs_path}.")

            db = None
            try:
                db = SessionLocal()
                project = Project(
                    project_name=project_name,
                    user_id=self.user_info.user_id,
                    project_abs_path=project_abs_path.as_posix(),
                )
                project = project_repository.save(db=db, model=project)

                return project

            except Exception as e:
                logger.error(f"Failed to save project '{project_name}' to the database: {e}")
                raise ProjectSaveException(error=e, project_name=project_name)
            finally:
                db and db.close()

    def get_projects(self) -> List[Project]:
        """
        Retrieve all projects associated with the current user.

        This method fetches project information from the database for
        the user identified by `self.user_info.user_id`.

        Returns:
            List[Project]: A list of projects associated with the current user.

        Raises:
            Exception: If an error occurs while querying the database.
        """
        db = None
        try:
            db = SessionLocal()
            projects = project_repository.get_all_by_user_id(db=db, user_id=self.user_info.user_id)

            return projects

        except Exception as e:
            logger.error(f"Failed to get project list from the database: {e}")
            raise
        finally:
            db and db.close()

    def get_project(self, project_id: str) -> Project:
        """
        Retrieve all projects associated with the current user.

        This method fetches project information from the database for
        the user identified by `self.user_info.user_id`.

        Returns:
            List[Project]: A list of projects associated with the current user.

        Raises:
            Exception: If an error occurs while querying the database.
        """
        db = None
        try:
            db = SessionLocal()
            project = project_repository.get_by_project_id(db=db, project_id=project_id)

            return project

        except Exception as e:
            logger.error(f"Failed to get project list from the database: {e}")
            raise
        finally:
            db and db.close()

    def trainer(self, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None) -> Trainer:
        """Initialize and return a Trainer instance.

        Args:
            task (Union[str, Task], optional): Type of task (classification, detection, segmentation).
            yaml_path (str, optional): Path to the YAML configuration file.

        Returns:
            Trainer: Initialized Trainer instance.
        """
        return Trainer(token_handler=self.token_handler, task=task, yaml_path=yaml_path)

    def compressor_v2(self) -> CompressorV2:
        """Initialize and return a Compressor instance.

        Returns:
            Compressor: Initialized Compressor instance.
        """
        return CompressorV2(token_handler=self.token_handler)

    def converter_v2(self) -> ConverterV2:
        """Initialize and return a Converter instance.

        Returns:
            Converter: Initialized Converter instance.
        """
        return ConverterV2(token_handler=self.token_handler, user_info=self.user_info)

    def quantizer(self) -> Quantizer:
        """Initialize and return a Quantizer instance.

        Returns:
            Quantizer: Initialized Quantizer instance.
        """
        return Quantizer(token_handler=self.token_handler, user_info=self.user_info)

    def benchmarker_v2(self) -> BenchmarkerV2:
        """Initialize and return a Benchmarker instance.

        Returns:
            Benchmarker: Initialized Benchmarker instance.
        """
        return BenchmarkerV2(token_handler=self.token_handler, user_info=self.user_info)

    def np_inferencer(self, config_path: str, input_model_path: str) -> NPInferencer:
        """Initialize and return a Inferencer instance.

        Returns:
            Inferencer: Initialized Inferencer instance.
        """

        return NPInferencer(config_path=config_path, input_model_path=input_model_path)

    def custom_inferencer(self, input_model_path: str) -> CustomInferencer:
        """Initialize and return a Inferencer instance.

        Returns:
            Inferencer: Initialized Inferencer instance.
        """
        return CustomInferencer(input_model_path=input_model_path)


class TAO:
    def __init__(self, ngc_api_key: str) -> None:
        """Initialize TAO instance and perform user authentication.

        Args:
            ngc_api_key (str): API key for TAO authentication.
        """
        self.ngc_api_key = ngc_api_key
        self.token_handler = TAOTokenHandler(ngc_api_key=ngc_api_key)

    def trainer(self) -> TAOTrainer:
        """Initialize and return a Trainer instance.

        Returns:
            TAO: Initialized Trainer instance.
        """
        return TAOTrainer(token_handler=self.token_handler)
