import os
import time
from pathlib import Path
from typing import Dict

from loguru import logger

from netspresso.clients.tao import tao_client
from netspresso.clients.utils.common import create_tao_headers
from netspresso.enums.action import ConvertAction, ExperimentAction
from netspresso.tao.models import MODELS
from netspresso.tao.utils.file import split_tar_file


class TAOTrainer:
    def __init__(self, ngc_api_key) -> None:
        self.ngc_api_key = ngc_api_key
        self.login()
        self.job_map = {}
        self.exp_map = {}

    def login(self):
        try:
            data = {"ngc_api_key": self.ngc_api_key}
            credentials = tao_client.auth.login(data)
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

    def upload_dataset(
        self, name: str, dataset_type: str, dataset_format: str, dataset_path: str, split_name: str = "train"
    ):
        try:
            logger.info("Creating dataset...")
            data = {"name": name, "type": dataset_type, "format": dataset_format}
            response = tao_client.dataset.create_dataset(self.user_id, data, self.headers)
            dataset_id = response["id"]

            dataset_path = Path(dataset_path)
            output_dir = dataset_path.parent / "split" / split_name
            split_tar_file(dataset_path, str(output_dir))

            for idx, tar_dataset_path in enumerate(output_dir.iterdir()):
                logger.info(f"Uploading {idx+1}/{len(list(output_dir.iterdir()))} tar split")
                upload_dataset_response = tao_client.dataset.upload_dataset(
                    self.user_id, dataset_id, str(tar_dataset_path), self.headers
                )
                logger.info(upload_dataset_response["message"])

            return response

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    def update_dataset(
        self, dataset_id: str, name: str = None, description: str = None, docker_env_vars: Dict[str, str] = None
    ):
        try:
            logger.info("Updating dataset...")
            data = {}
            if name:
                data["name"] = name
            if description:
                data["description"] = description
            if docker_env_vars:
                data["docker_env_vars"] = docker_env_vars
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

    def run_dataset_jobs(self, dataset_id, parent_job_id, action, specs):
        try:
            logger.info("Running dataset jobs...")
            data = {"parent_job_id": parent_job_id, "action": action, "specs": specs}
            response = tao_client.dataset.run_dataset_jobs(self.user_id, dataset_id, data, self.headers)

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

    def get_dataset_schema(self, dataset_id, action):
        try:
            logger.info("Getting dataset schema...")
            response = tao_client.dataset.get_specs_schema(self.user_id, dataset_id, action, self.headers)

            return response

        except Exception as e:
            logger.error(f"Convert dataset failed. Error: {e}")
            raise e

    def create_experiment(self, network_arch, encryption_key, checkpoint_choose_method):
        try:
            logger.info("Creating experiment...")
            data = {
                "network_arch": network_arch,
                "encryption_key": encryption_key,
                "checkpoint_choose_method": checkpoint_choose_method,
            }
            response = tao_client.experiment.create_experiments(self.user_id, data, self.headers)

            return response

        except Exception as e:
            logger.error(f"Create experiment failed. Error: {e}")
            raise e

    def assign_datasets(self, experiment_id, train_datasets, eval_dataset, inference_dataset, calibration_dataset):
        try:
            logger.info("Assigning datasets...")
            data = {
                "train_datasets": train_datasets,
                "eval_dataset": eval_dataset,
                "inference_dataset": inference_dataset,
                "calibration_dataset": calibration_dataset,
            }
            response = tao_client.experiment.partial_update_experiment(self.user_id, experiment_id, data, self.headers)

            return response

        except Exception as e:
            logger.error(f"Assign datasets failed. Error: {e}")
            raise e

    def assign_ptm(self, experiment_id, network_arch, ptm_name):
        try:
            logger.info("Assigning pre-trained model...")
            no_ptm_models = set()

            # Get pretrained model for entered task
            if network_arch not in no_ptm_models:
                experiments = tao_client.experiment.get_experiments(
                    user_id=self.user_id, headers=self.headers, network_arch=network_arch
                )

                # Search for ptm with given ngc path
                ptm = []
                for experiment in experiments:
                    rsp_keys = experiment.keys()
                    assert "ngc_path" in rsp_keys
                    if experiment["ngc_path"].endswith(MODELS[network_arch][ptm_name]):
                        logger.info(f"PTM info: {experiment}")
                        assert "id" in rsp_keys
                        ptm = [experiment["id"]]
                        logger.info("Metadata for model with requested NGC Path")
                        break

                data = {"base_experiment": ptm}
                response = tao_client.experiment.partial_update_experiment(
                    self.user_id, experiment_id, data, self.headers
                )

            return response

        except Exception as e:
            logger.error(f"Assign pre-trained model failed. Error: {e}")
            raise e

    def get_experiments(
        self, skip=None, size=None, sort=None, name=None, type=None, network_arch=None, read_only=None, user_only=None
    ):
        try:
            logger.info("Getting experiments...")
            experiments = tao_client.experiment.get_experiments(
                self.user_id,
                self.headers,
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
            experiment = tao_client.experiment.get_experiment(self.user_id, experiment_id, self.headers)

            return experiment

        except Exception as e:
            logger.error(f"Get experiment failed. Error: {e}")
            raise e

    def update_experiment(self, experiment_id, name):
        try:
            logger.info("Updating experiment...")
            data = {"name": name}
            response = tao_client.experiment.partial_update_experiment(self.user_id, experiment_id, data, self.headers)

            return response

        except Exception as e:
            logger.error(f"Update experiment failed. Error: {e}")
            raise e

    def delete_experiment(self, experiment_id):
        try:
            logger.info("Deleting experiment...")
            response = tao_client.experiment.delete_experiment(self.user_id, experiment_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Delete experiment failed. Error: {e}")
            raise e

    def get_train_schema(self, experiment_id):
        try:
            logger.info("Getting train schema...")
            response = tao_client.experiment.get_specs_schema(
                self.user_id, experiment_id, ExperimentAction.TRAIN, self.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get train schema failed. Error: {e}")
            raise e

    def train(self, experiment_id, name, train_specs, parent_job_id=None):
        try:
            logger.info("Running train...")
            data = {
                "name": name,
                "action": ExperimentAction.TRAIN,
                "parent_job_id": parent_job_id,
                "specs": train_specs,
            }
            train_job_id = tao_client.experiment.run_experiment_jobs(self.user_id, experiment_id, data, self.headers)

            return train_job_id

        except Exception as e:
            logger.error(f"Train failed. Error: {e}")
            raise e

    def cancel_experiment_job(self, experiment_id, job_id):
        try:
            logger.info("Canceling experiment job...")
            response = tao_client.experiment.cancel_experiment_job(self.user_id, experiment_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Cancel experiment job failed. Error: {e}")
            raise e

    def get_experiment_jobs(self, experiment_id):
        try:
            logger.info("Getting experiment jobs...")
            response = tao_client.experiment.get_experiment_jobs(self.user_id, experiment_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Get experiment jobs failed. Error: {e}")
            raise e

    def get_experiment_job(self, experiment_id, job_id):
        try:
            logger.info("Getting experiment job...")
            response = tao_client.experiment.get_experiment_job(self.user_id, experiment_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Get experiment job failed. Error: {e}")
            raise e

    def delete_experiment_job(self, experiment_id, job_id):
        try:
            logger.info("Deleting experiment job...")
            response = tao_client.experiment.delete_experiment_job(self.user_id, experiment_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Delete experiment job failed. Error: {e}")
            raise e

    def resume_experiment_job(self, experiment_id, job_id):
        try:
            logger.info("Resuming experiment job...")
            response = tao_client.experiment.resume_experiment_job(self.user_id, experiment_id, job_id, self.headers)

            return response

        except Exception as e:
            logger.error(f"Resume experiment job failed. Error: {e}")
            raise e

    def get_evaluate_schema(self, experiment_id):
        try:
            logger.info("Getting evaluate schema...")
            response = tao_client.experiment.get_specs_schema(
                self.user_id, experiment_id, ExperimentAction.EVALUATE, self.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get evaluate schema failed. Error: {e}")
            raise e

    def evaluate(self, experiment_id, eval_specs, parent_job_id):
        try:
            logger.info("Evaluating...")
            data = {
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.EVALUATE,
                "specs": eval_specs,
            }
            evaluate_job_id = tao_client.experiment.run_experiment_jobs(self.user_id, experiment_id, data, self.headers)

            return evaluate_job_id

        except Exception as e:
            logger.error(f"Evaluate failed. Error: {e}")
            raise e

    def get_export_schema(self, experiment_id):
        try:
            logger.info("Getting export schema...")
            response = tao_client.experiment.get_specs_schema(
                self.user_id, experiment_id, ExperimentAction.EXPORT, self.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get export schema failed. Error: {e}")
            raise e

    def export(self, experiment_id, export_specs, parent_job_id):
        try:
            logger.info("Exporting...")
            data = {
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.EXPORT,
                "specs": export_specs,
            }
            export_job_id = tao_client.experiment.run_experiment_jobs(self.user_id, experiment_id, data, self.headers)

            return export_job_id

        except Exception as e:
            logger.error(f"Export failed. Error: {e}")
            raise e

    def get_gen_trt_engine_schema(self, experiment_id):
        try:
            logger.info("Getting gen_trt_engine schema...")
            response = tao_client.experiment.get_specs_schema(
                self.user_id, experiment_id, ExperimentAction.GEN_TRT_ENGINE, self.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get gen_trt_engine schema failed. Error: {e}")
            raise e

    def gen_trt_engine(self, experiment_id, tao_deploy_specs, parent_job_id):
        try:
            logger.info("Generating trt engine...")
            data = {
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.GEN_TRT_ENGINE,
                "specs": tao_deploy_specs,
            }
            gen_trt_job_id = tao_client.experiment.run_experiment_jobs(self.user_id, experiment_id, data, self.headers)

            return gen_trt_job_id

        except Exception as e:
            logger.error(f"Generate trt engine failed. Error: {e}")
            raise e

    def get_inference_schema(self, experiment_id):
        try:
            logger.info("Getting inference schema...")
            response = tao_client.experiment.get_specs_schema(
                self.user_id, experiment_id, ExperimentAction.INFERENCE, self.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get inference schema failed. Error: {e}")
            raise e

    def inference(self, experiment_id, inference_specs, parent_job_id):
        try:
            logger.info("Inferring...")
            data = {
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.INFERENCE,
                "specs": inference_specs,
            }
            inference_job_id = tao_client.experiment.run_experiment_jobs(
                self.user_id, experiment_id, data, self.headers
            )

            return inference_job_id

        except Exception as e:
            logger.error(f"Inference failed. Error: {e}")
            raise e

    def download_artifacts(self, experiment_id, job_id, output_dir, best_model=False, latest_model=True):
        try:
            logger.info("Downloading selective artifacts...")
            file_lists = tao_client.experiment.get_list_files(self.user_id, experiment_id, job_id, self.headers)
            logger.info(f"File lists: {file_lists}")

            # Save
            temptar = f"{job_id}.tar.gz"
            with tao_client.experiment.download_selective_files(
                self.user_id, experiment_id, job_id, file_lists, best_model, latest_model, self.headers
            ) as r:
                r.raise_for_status()
                with open(temptar, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Untar to destination
            logger.info("Untarring...")
            Path(output_dir).mkdir(exist_ok=True)
            tar_command = f"tar -xvf {temptar} -C {output_dir}/"
            os.system(tar_command)
            os.remove(temptar)

            return output_dir

        except Exception as e:
            logger.error(f"Download artifacts failed. Error: {e}")
            raise e

    def monitor_job_status(self, experiment_id, job_id, interval=15):
        try:
            logger.info("Monitoring job stauts...")

            while True:
                response = tao_client.experiment.get_experiment_job(self.user_id, experiment_id, job_id, self.headers)
                logger.info(response)
                if response.get("status") in ["Done", "Error", "Canceled"]:
                    break
                time.sleep(interval)
            return response

        except Exception as e:
            logger.error(f"Monitor job staus failed. Error: {e}")
            raise e

        except KeyboardInterrupt:
            logger.info("End monitoring.")
