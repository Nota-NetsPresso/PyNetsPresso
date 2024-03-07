from loguru import logger

from netspresso.clients.tao import tao_client
from netspresso.enums.tao.action import ConvertAction


class Dataset:
    def __init__(self, id, token_handler) -> None:
        self.id = id
        self.token_handler = token_handler

    def update_dataset(self, dataset_id: str, name: str = None, description: str = None):
        try:
            logger.info("Updating dataset...")
            data = {}
            if name:
                data["name"] = name
            if description:
                data["description"] = description
            response = tao_client.dataset.update_dataset(
                self.token_handler.user_id, dataset_id, data, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Update dataset failed. Error: {e}")
            raise e

    def delete_dataset(self, dataset_id: str):
        try:
            logger.info("Deleting dataset...")
            response = tao_client.dataset.delete_dataset(
                self.token_handler.user_id, dataset_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Delete dataset failed. Error: {e}")
            raise e

    def convert_dataset(self, dataset_id: str, action: ConvertAction):
        try:
            logger.info("Converting dataset...")
            response = tao_client.dataset.get_specs_schema(
                self.token_handler.user_id, dataset_id, action, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Convert dataset failed. Error: {e}")
            raise e

    def get_dataset_schema(self, dataset_id, action):
        try:
            logger.info("Getting dataset schema...")
            response = tao_client.dataset.get_specs_schema(
                self.token_handler.user_id, dataset_id, action, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Convert dataset failed. Error: {e}")
            raise e

    def run_dataset_jobs(self, parent_job_id, action, specs):
        try:
            logger.info("Running dataset jobs...")
            data = {"parent_job_id": parent_job_id, "action": action, "specs": specs}
            response = tao_client.dataset.run_dataset_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Run dataset jobs failed. Error: {e}")
            raise e

    def get_dataset_jobs(self, skip=None, size=None, sort=None):
        try:
            logger.info("Getting dataset jobs...")
            response = tao_client.dataset.get_dataset_jobs(
                self.token_handler.user_id, self.id, self.token_handler.headers, skip, size, sort
            )

            return response

        except Exception as e:
            logger.error(f"Get dataset jobs failed. Error: {e}")
            raise e

    def get_dataset_job(self, job_id: str):
        try:
            logger.info("Getting dataset job...")
            response = tao_client.dataset.get_dataset_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get dataset job. Error: {e}")
            raise e

    def delete_dataset_job(self, job_id: str):
        try:
            logger.info("Deleting dataset job...")
            response = tao_client.dataset.delete_dataset_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Delete dataset job failed. Error: {e}")
            raise e

    def cancel_dataset_job(self, job_id: str):
        try:
            logger.info("Canceling dataset job...")
            response = tao_client.dataset.cancel_dataset_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Cancel dataset job failed. Error: {e}")
            raise e

    def download_job_artifacts(self, job_id: str):
        try:
            logger.info("Downloading job artifacts...")
            response = tao_client.dataset.download_job_artifacts(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Download job artifacts failed. Error: {e}")
            raise e
