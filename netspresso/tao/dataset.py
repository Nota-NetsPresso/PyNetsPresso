import time

from loguru import logger

from netspresso.clients.tao import tao_client
from netspresso.enums.tao.action import ConvertAction


class Dataset:
    def __init__(self, id, token_handler) -> None:
        self.id = id
        self.token_handler = token_handler
        self.job_map = {}
        self.convert_job_cnt = 1

    def update_dataset(self, name: str = None, description: str = None):
        try:
            logger.info("Updating dataset...")
            data = {}
            if name:
                data["name"] = name
            if description:
                data["description"] = description
            response = tao_client.dataset.update_dataset(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Update dataset failed. Error: {e}")
            raise e

    def delete_dataset(self):
        try:
            logger.info("Deleting dataset...")
            response = tao_client.dataset.delete_dataset(
                self.token_handler.user_id, self.id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Delete dataset failed. Error: {e}")
            raise e

    def get_dataset_convert_specs(self, action: ConvertAction):
        try:
            logger.info("Getting dataset convert specs...")
            response = tao_client.dataset.get_specs_schema(
                self.token_handler.user_id, self.id, action, self.token_handler.headers
            )
            dataset_convert_specs = response["default"]

            return dataset_convert_specs

        except Exception as e:
            logger.error(f"Convert dataset failed. Error: {e}")
            raise e

    def convert_dataset(self, parent_job_id, action: ConvertAction, specs):
        try:
            logger.info("Converting dataset...")
            data = {"parent_job_id": parent_job_id, "action": action, "specs": specs}
            convert_job_id = tao_client.dataset.run_dataset_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )
            job_key = f"convert_job_{self.convert_job_cnt}"
            self.job_map[job_key] = convert_job_id
            self.convert_job_cnt += 1

            return convert_job_id

        except Exception as e:
            logger.error(f"Convert dataset failed. Error: {e}")
            raise e

    def monitor_job_status(self, job_id, interval=15):
        try:
            logger.info("Monitoring dataset job stauts...")

            while True:
                response = tao_client.dataset.get_dataset_job(
                    self.token_handler.user_id, self.id, job_id, self.token_handler.headers
                )
                logger.info(response)
                if response.get("status") in ["Done", "Error", "Canceled"]:
                    break
                time.sleep(interval)

        except Exception as e:
            logger.error(f"Monitor dataset job staus failed. Error: {e}")
            raise e

        except KeyboardInterrupt:
            logger.info("End monitoring.")

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
