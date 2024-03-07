from pathlib import Path

from loguru import logger

from netspresso.clients.tao import TAOTokenHandler, tao_client
from netspresso.enums.tao.experiment import CheckpointChooseMethod, EncryptionKey, NetworkArch
from netspresso.tao.dataset import Dataset
from netspresso.tao.experiment import Experiment
from netspresso.tao.utils.file import split_tar_file


class TAOTrainer:
    def __init__(self, token_handler: TAOTokenHandler) -> None:
        self.token_handler = token_handler

    def upload_dataset(
        self, name: str, dataset_type: str, dataset_format: str, dataset_path: str, split_name: str = "train"
    ):
        try:
            logger.info("Creating dataset...")
            data = {"name": name, "type": dataset_type, "format": dataset_format}
            response = tao_client.dataset.create_dataset(self.token_handler.user_id, data, self.token_handler.headers)
            dataset_id = response["id"]

            dataset_path = Path(dataset_path)
            output_dir = dataset_path.parent / "split" / split_name
            split_tar_file(dataset_path, str(output_dir))

            for idx, tar_dataset_path in enumerate(output_dir.iterdir()):
                logger.info(f"Uploading {idx+1}/{len(list(output_dir.iterdir()))} tar split")
                upload_dataset_response = tao_client.dataset.upload_dataset(
                    self.token_handler.user_id, dataset_id, str(tar_dataset_path), self.token_handler.headers
                )
                logger.info(upload_dataset_response["message"])

            dataset = Dataset(id=response["id"], token_handler=self.token_handler)

            return dataset

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    def get_datasets(self, skip=None, size=None, sort=None, name=None, format=None, type=None):
        try:
            logger.info("Getting datasets...")
            response = tao_client.dataset.get_datasets(
                self.token_handler.user_id, self.token_handler.headers, skip, size, sort, name, format, type
            )

            return response

        except Exception as e:
            logger.error(f"Get datasets failed. Error: {e}")
            raise e

    def get_dataset(self, dataset_id: str):
        try:
            logger.info("Getting dataset...")
            response = tao_client.dataset.get_dataset(
                self.token_handler.user_id, dataset_id, self.token_handler.headers
            )

            dataset = Dataset(id=response["id"], token_handler=self.token_handler)

            return dataset

        except Exception as e:
            logger.error(f"Get dataset failed. Error: {e}")
            raise e

    def create_experiment(
        self,
        name: str,
        network_arch: NetworkArch,
        encryption_key: EncryptionKey,
        checkpoint_choose_method: CheckpointChooseMethod = CheckpointChooseMethod.BEST_MODEL,
    ):
        try:
            logger.info("Creating experiment...")
            data = {
                "name": name,
                "network_arch": network_arch,
                "encryption_key": encryption_key,
                "checkpoint_choose_method": checkpoint_choose_method,
            }
            response = tao_client.experiment.create_experiments(
                self.token_handler.user_id, data, self.token_handler.headers
            )
            experiment = Experiment(
                name=name, id=response["id"], network_arch=network_arch, token_handler=self.token_handler
            )
            logger.info(f"Created experiment. experiment id: {experiment.id}")

            return experiment

        except Exception as e:
            logger.error(f"Create experiment failed. Error: {e}")
            raise e

    def get_experiments(
        self, skip=None, size=None, sort=None, name=None, type=None, network_arch=None, read_only=None, user_only=None
    ):
        try:
            logger.info("Getting experiments...")
            experiments = tao_client.experiment.get_experiments(
                self.token_handler.user_id,
                self.token_handler.headers,
                skip,
                size,
                sort,
                name,
                type,
                network_arch,
                read_only,
                user_only,
            )

            return experiments

        except Exception as e:
            logger.error(f"Get train schema failed. Error: {e}")
            raise e

    def get_experiment(self, experiment_id):
        try:
            logger.info("Getting experiment...")
            response = tao_client.experiment.get_experiment(
                self.token_handler.user_id, experiment_id, self.token_handler.headers
            )
            experiment = Experiment(
                name=response["name"],
                id=response["id"],
                network_arch=response["network_arch"],
                token_handler=self.token_handler,
            )

            return experiment

        except Exception as e:
            logger.error(f"Get experiment failed. Error: {e}")
            raise e
