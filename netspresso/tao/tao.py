from pathlib import Path
from typing import Dict

from loguru import logger

from netspresso.tao.utils.file import split_tar_file
from netspresso.clients.tao import tao_client
from netspresso.clients.utils.common import create_tao_headers
from netspresso.enums.action import ConvertAction


class TAO:
    def __init__(self, ngc_api_key) -> None:
        self.ngc_api_key = ngc_api_key
        self.login()

    def login(self):
        try:
            credentials = tao_client.auth.login({"ngc_api_key": self.ngc_api_key})
            logger.info("Login was successfully to TAO.")
            self.user_id = credentials["user_id"]
            self.token = credentials["token"]
            self.headers = create_tao_headers(self.token)

        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    def create_dataset(self, dataset_type: str, dataset_format: str):
        try:
            logger.info("Creating dataset...")
            data = {"type": dataset_type, "format": dataset_format}
            response = tao_client.dataset.create_dataset(self.user_id, data, self.headers)

            return response

        except Exception as e:
            logger.error(f"Create dataset failed. Error: {e}")
            raise e

    def upload_dataset(self, name: str, dataset_type: str, dataset_format: str, dataset_path: str, split_name: str = "train"):
        try:
            logger.info("Creating dataset...")
            data = {"name": name, "type": dataset_type, "format": dataset_format}
            response = tao_client.dataset.create_dataset(self.user_id, data, self.headers)
            dataset_id = response["id"]

            dataset_path = Path(dataset_path)
            output_dir = dataset_path.parent / split_name
            split_tar_file(dataset_path, str(output_dir))

            for idx, tar_dataset_path in enumerate(output_dir.iterdir()):
                logger.info(f"Uploading {idx+1}/{len(list(output_dir.iterdir()))} tar split")
                upload_dataset_response = tao_client.dataset.upload_dataset(self.user_id, dataset_id, str(tar_dataset_path), self.headers)
                logger.info(upload_dataset_response["message"])

            return response

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    def update_dataset(self, dataset_id: str, name: str = None, description: str = None, docker_env_vars: Dict[str, str] = {}):
        try:
            logger.info("Updating dataset...")
            data = {"docker_env_vars": docker_env_vars}
            if name:
                data["name"] = name
            if description:
                data["description"] = description
            response = tao_client.dataset.update_dataset(self.user_id, dataset_id, data, self.headers)

            return response

        except Exception as e:
            logger.error(f"Update dataset failed. Error: {e}")
            raise e

    def delete_dataset(self, dataset_id: str):
        try:
            logger.info("Deleting dataset...")
            response = tao_client.dataset.delete_dataset(self.user_id, dataset_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Delete dataset failed. Error: {e}")
            raise e

    def get_datasets(self, skip=None, size=None, sort=None, name=None, format=None, type=None):
        try:
            logger.info("Getting datasets...")
            response = tao_client.dataset.get_datasets(self.user_id, self.headers, skip, size, sort, name, format, type)
            
            return response

        except Exception as e:
            logger.error(f"Get datasets failed. Error: {e}")
            raise e

    def get_dataset(self, dataset_id: str):
        try:
            logger.info("Getting dataset...")
            response = tao_client.dataset.get_dataset(self.user_id, dataset_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Get dataset failed. Error: {e}")
            raise e

    def get_dataset_jobs(self, dataset_id: str, skip=None, size=None, sort=None):
        try:
            logger.info("Getting dataset jobs...")
            response = tao_client.dataset.get_dataset_jobs(self.user_id, dataset_id, self.headers, skip, size, sort)

            return response

        except Exception as e:
            logger.error(f"Get dataset jobs failed. Error: {e}")
            raise e

    def run_dataset_jobs(self, dataset_id: str, job_id: str):
        try:
            logger.info("Running dataset jobs...")
            response = tao_client.dataset.get_dataset_job(self.user_id, dataset_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Run dataset jobs failed. Error: {e}")
            raise e

    def delete_dataset_job(self, dataset_id: str, job_id: str):
        try:
            logger.info("Deleting dataset job...")
            response = tao_client.dataset.delete_dataset_job(self.user_id, dataset_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Delete dataset job failed. Error: {e}")
            raise e

    def get_dataset_job(self, dataset_id: str, job_id: str):
        try:
            logger.info("Getting dataset job...")
            response = tao_client.dataset.get_dataset_job(self.user_id, dataset_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Get dataset job. Error: {e}")
            raise e

    def cancel_dataset_job(self, dataset_id: str, job_id: str):
        try:
            logger.info("Canceling dataset job...")
            response = tao_client.dataset.cancel_dataset_job(self.user_id, dataset_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Cancel dataset job failed. Error: {e}")
            raise e

    def download_job_artifacts(self, dataset_id: str, job_id: str):
        try:
            logger.info("Downloading job artifacts...")
            response = tao_client.dataset.download_job_artifacts(self.user_id, dataset_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Download job artifacts failed. Error: {e}")
            raise e

    def convert_dataset(self, dataset_id: str, action: ConvertAction):
        try:
            logger.info("Converting dataset...")
            response = tao_client.dataset.get_specs_schema(self.user_id, dataset_id, action, self.headers)

            return response

        except Exception as e:
            logger.error(f"Convert dataset failed. Error: {e}")
            raise e
