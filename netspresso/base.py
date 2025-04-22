from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.enums import ServiceCredit, ServiceTask, Status
from netspresso.exceptions.common import NotEnoughCreditException
from netspresso.metadata.common import BaseMetadata
from netspresso.utils.db.repositories.project import project_repository
from netspresso.utils.db.session import get_db_session


class NetsPressoBase:
    def __init__(self, token_handler: TokenHandler) -> None:
        self.token_handler = token_handler
        self.auth_client = auth_client

    def check_credit_balance(self, service_task: ServiceTask):
        current_credit = self.auth_client.get_credit(
            access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
        )
        service_credit = ServiceCredit.get_credit(service_task)
        service_task_name = service_task.name.replace("_", " ").lower()
        if current_credit < service_credit:
            logger.error(
                f"Insufficient balance: {current_credit} credits available, but {service_credit} credits required for {service_task_name} task."
            )
            raise NotEnoughCreditException(current_credit, service_credit, service_task_name)

    def print_remaining_credit(self, service_task):
        if self.auth_client.is_cloud():
            self.token_handler.validate_token()
            service_credit = ServiceCredit.get_credit(service_task)
            remaining_credit = self.auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(f"{service_credit} credits have been consumed. Remaining Credit: {remaining_credit}")

    def validate_token_and_check_credit(self, service_task: ServiceTask):
        self.token_handler.validate_token()
        self.check_credit_balance(service_task=service_task)

    def handle_error(self, metadata: BaseMetadata, task_name: ServiceTask, error_message: str):
        metadata.status = Status.ERROR
        metadata.update_message(exception_detail=error_message)
        logger.error(f"{task_name} task failed due to an error: {error_message}")

        return metadata

    def handle_stop(self, metadata: BaseMetadata, task_name: ServiceTask):
        metadata.status = Status.STOPPED
        logger.error(f"{task_name} task was interrupted by the user.")

        return metadata

    def get_project(self, project_id):
        with get_db_session() as db:
            project = project_repository.get_by_project_id(db=db, project_id=project_id)

            return project
