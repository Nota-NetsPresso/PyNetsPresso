import os
import time
from pathlib import Path
from typing import List

from loguru import logger

from netspresso.clients.tao import tao_client
from netspresso.enums.tao.action import ExperimentAction


class Experiment:
    def __init__(self, id, name, network_arch, token_handler) -> None:
        self.id = id
        self.name = name
        self.network_arch = network_arch
        self.token_handler = token_handler
        self.pretrained_model = None
        self.data = None
        self.train_specs = self.get_train_specs()
        self.export_specs = self.get_export_specs()
        self.evaluate_specs = self.get_evaluate_specs()
        self.job_map = {}
        self.train_job_cnt = 1
        self.export_job_cnt = 1

    def delete_experiment(self):
        try:
            logger.info("Deleting experiment...")
            response = tao_client.experiment.delete_experiment(
                self.token_handler.user_id, self.id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Delete experiment failed. Error: {e}")
            raise e

    def set_dataset(
        self, train_datasets: List[str], eval_dataset: str, inference_dataset: str, calibration_dataset: str
    ):
        self.data = {
            "train_datasets": train_datasets,
            "eval_dataset": eval_dataset,
            "inference_dataset": inference_dataset,
            "calibration_dataset": calibration_dataset,
        }

    def set_pretrained_model(self, pretrained_model_name: str):
        # Check if network architecture requires a pretrained model
        # Get experiments for the given network architecture
        experiments = tao_client.experiment.get_experiments(
            user_id=self.token_handler.user_id, headers=self.token_handler.headers, network_arch=self.network_arch
        )

        # Search for pretrained model with the given NGC path
        for experiment in experiments:
            ngc_path = experiment.get("ngc_path")
            if ngc_path and ngc_path.endswith(pretrained_model_name):
                print(ngc_path)
                logger.info(f"Pretrained model info: {experiment}")
                ptm_id = experiment.get("id")
                logger.info("Metadata for model with requested NGC Path")
                self.pretrained_model = {"base_experiment": [ptm_id]}
                break

        else:
            logger.warning("No pretrained model specified for the given PTM name.")

    def set_automl_with_bayesian(self, metric, automl_max_recommendations, override_automl_disabled_params, additional_automl_params, remove_default_automl_params):
        automl_information = {
            "automl_enabled": True,
            "automl_algorithm": "bayesian",
            "metric": metric,
            "automl_max_recommendations": automl_max_recommendations,
            "override_automl_disabled_params": override_automl_disabled_params,
            "automl_add_hyperparameters": str(additional_automl_params),
            "automl_remove_hyperparameters": str(remove_default_automl_params)
        }

        tao_client.experiment.partial_update_experiment(
            self.token_handler.user_id, self.id, automl_information, self.token_handler.headers
        )

    def set_automl_with_hyperband(self, metric, automl_R, automl_nu, epoch_multiplier, override_automl_disabled_params, additional_automl_params, remove_default_automl_params):
        automl_information = {
            "automl_enabled": True,
            "automl_algorithm": "hyperband",
            "metric": metric,
            "automl_R": automl_R,
            "automl_nu": automl_nu,
            "epoch_multiplier": epoch_multiplier,
            "override_automl_disabled_params": override_automl_disabled_params,
            "automl_add_hyperparameters": str(additional_automl_params),
            "automl_remove_hyperparameters": str(remove_default_automl_params)
        }

        tao_client.experiment.partial_update_experiment(
            self.token_handler.user_id, self.id, automl_information, self.token_handler.headers
        )

    def get_train_specs(self):
        try:
            logger.info("Getting train specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.TRAIN, self.token_handler.headers
            )
            train_specs = response["default"]

            return train_specs

        except Exception as e:
            logger.error(f"Get train specs failed. Error: {e}")
            raise e

    def get_export_specs(self):
        try:
            logger.info("Getting export specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.EXPORT, self.token_handler.headers
            )
            export_specs = response["default"]

            return export_specs

        except Exception as e:
            logger.error(f"Get export specs failed. Error: {e}")
            raise e

    def get_evaluate_specs(self):
        try:
            logger.info("Getting evaluate specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.EVALUATE, self.token_handler.headers
            )
            evaluate_specs = response["default"]

            return evaluate_specs

        except Exception as e:
            logger.error(f"Get evaluate specs failed. Error: {e}")
            raise e

    def get_prune_specs(self):
        try:
            logger.info("Getting prune specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.PRUNE, self.token_handler.headers
            )
            prune_specs = response["default"]

            return prune_specs

        except Exception as e:
            logger.error(f"Get prune specs failed. Error: {e}")
            raise e

    def get_retrain_specs(self):
        try:
            logger.info("Getting retrain specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.RETRAIN, self.token_handler.headers
            )
            retrain_specs = response["default"]

            return retrain_specs

        except Exception as e:
            logger.error(f"Get retrain specs failed. Error: {e}")
            raise e

    def get_trt_engine_spces(self):
        try:
            logger.info("Getting trt engine specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.GEN_TRT_ENGINE, self.token_handler.headers
            )
            trt_engine_specs = response["default"]

            return trt_engine_specs

        except Exception as e:
            logger.error(f"Get trt engine specs failed. Error: {e}")
            raise e

    def get_inference_spces(self):
        try:
            logger.info("Getting inference specs...")
            response = tao_client.experiment.get_specs_schema(
                self.token_handler.user_id, self.id, ExperimentAction.INFERENCE, self.token_handler.headers
            )
            inference_specs = response["default"]

            return inference_specs

        except Exception as e:
            logger.error(f"Get inference specs failed. Error: {e}")
            raise e

    def train(self, name: str, parent_job_id: str = None):
        try:
            logger.info("Running train...")
            update_data = {**self.data, **self.pretrained_model}
            tao_client.experiment.partial_update_experiment(
                self.token_handler.user_id, self.id, update_data, self.token_handler.headers
            )

            data = {
                "name": name,
                "action": ExperimentAction.TRAIN,
                "parent_job_id": parent_job_id,
                "specs": self.train_specs,
            }
            train_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )
            job_key = f"train_job_{self.train_job_cnt}"
            self.job_map[job_key] = train_job_id
            self.train_job_cnt += 1

            return train_job_id

        except Exception as e:
            logger.error(f"Train failed. Error: {e}")
            raise e

    def evaluate(self, parent_job_id):
        try:
            logger.info("Evaluating...")
            data = {
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.EVALUATE,
                "specs": self.eval_specs,
            }
            evaluate_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return evaluate_job_id

        except Exception as e:
            logger.error(f"Evaluate failed. Error: {e}")
            raise e

    def export(self, parent_job_id):
        try:
            logger.info("Exporting...")
            data = {
                "name": f"{self.name} export",
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.EXPORT,
                "specs": self.export_specs,
            }
            export_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )
            job_key = f"export_job_{self.export_job_cnt}"
            self.job_map[job_key] = export_job_id
            self.export_job_cnt += 1

            return export_job_id

        except Exception as e:
            logger.error(f"Export failed. Error: {e}")
            raise e

    def prune(self, parent_job_id, prune_specs):
        try:
            logger.info("Pruning...")
            data = {
                "name": f"{self.name} prune",
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.PRUNE,
                "specs": prune_specs,
            }
            prune_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return prune_job_id

        except Exception as e:
            logger.error(f"Prune failed. Error: {e}")
            raise e

    def retrain(self, name: str, parent_job_id: str, retrain_specs):
        try:
            logger.info("Running train...")
            update_data = {**self.data, **self.pretrained_model}
            tao_client.experiment.partial_update_experiment(
                self.token_handler.user_id, self.id, update_data, self.token_handler.headers
            )

            data = {
                "name": name,
                "action": ExperimentAction.RETRAIN,
                "parent_job_id": parent_job_id,
                "specs": retrain_specs,
            }
            retrain_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return retrain_job_id

        except Exception as e:
            logger.error(f"Retrain failed. Error: {e}")
            raise e

    def gen_trt_engine(self, parent_job_id, trt_engine_specs):
        try:
            logger.info("Generating TRT engine...")
            data = {
                "name": f"{self.name} trt engine",
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.GEN_TRT_ENGINE,
                "specs": trt_engine_specs,
            }
            gen_trt_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return gen_trt_job_id

        except Exception as e:
            logger.error(f"Generate TRT engine failed. Error: {e}")
            raise e

    def inference(self, parent_job_id, inference_specs):
        try:
            logger.info("Inferring...")
            data = {
                "name": f"{self.name} inference",
                "parent_job_id": parent_job_id,
                "action": ExperimentAction.INFERENCE,
                "specs": inference_specs,
            }
            inference_job_id = tao_client.experiment.run_experiment_jobs(
                self.token_handler.user_id, self.id, data, self.token_handler.headers
            )

            return inference_job_id

        except Exception as e:
            logger.error(f"Inference failed. Error: {e}")
            raise e

    def monitor_job_status(self, job_id, interval=15):
        try:
            logger.info("Monitoring experiment job stauts...")

            while True:
                response = tao_client.experiment.get_experiment_job(
                    self.token_handler.user_id, self.id, job_id, self.token_handler.headers
                )
                logger.info(response)
                if response.get("status") in ["Done", "Error", "Canceled"]:
                    break
                time.sleep(interval)

        except Exception as e:
            logger.error(f"Monitor experiment job staus failed. Error: {e}")
            raise e

        except KeyboardInterrupt:
            logger.info("End monitoring.")

    def get_experiment_jobs(self):
        try:
            logger.info("Getting experiment jobs...")
            response = tao_client.experiment.get_experiment_jobs(
                self.token_handler.user_id, self.id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get experiment jobs failed. Error: {e}")
            raise e

    def get_experiment_job(self, job_id):
        try:
            logger.info("Getting experiment job...")
            response = tao_client.experiment.get_experiment_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Get experiment job failed. Error: {e}")
            raise e

    def delete_experiment_job(self, job_id):
        try:
            logger.info("Deleting experiment job...")
            response = tao_client.experiment.delete_experiment_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Delete experiment job failed. Error: {e}")
            raise e

    def resume_experiment_job(self, job_id):
        try:
            logger.info("Resuming experiment job...")
            response = tao_client.experiment.resume_experiment_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Resume experiment job failed. Error: {e}")
            raise e

    def cancel_experiment_job(self, job_id):
        try:
            logger.info("Canceling experiment job...")
            response = tao_client.experiment.cancel_experiment_job(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )

            return response

        except Exception as e:
            logger.error(f"Cancel experiment job failed. Error: {e}")
            raise e

    def download_artifacts(self, job_id, output_dir, best_model=True, latest_model=False):
        try:
            logger.info("Downloading selective artifacts...")
            file_lists = tao_client.experiment.get_list_files(
                self.token_handler.user_id, self.id, job_id, self.token_handler.headers
            )
            logger.info(f"File lists: {file_lists}")

            # Save
            temptar = f"{job_id}.tar.gz"
            with tao_client.experiment.download_selective_files(
                self.token_handler.user_id,
                self.id,
                job_id,
                file_lists,
                best_model,
                latest_model,
                self.token_handler.headers,
            ) as r:
                r.raise_for_status()
                with open(temptar, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Untar to destination
            logger.info("Untarring...")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            tar_command = f"tar -xvf {temptar} -C {output_dir}/"
            os.system(tar_command)
            os.remove(temptar)

            return output_dir

        except Exception as e:
            logger.error(f"Download artifacts failed. Error: {e}")
            raise e
