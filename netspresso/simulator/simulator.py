from pathlib import Path
from typing import Optional

from loguru import logger

from netspresso.analytics import netspresso_analytics
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.model.response_body import ResponseModelItem
from netspresso.enums import ServiceTask, Status
from netspresso.enums.simulate import SimulateTaskType
from netspresso.metadata.simulator import SimulatorMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class Simulator(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Simulator."""

        super().__init__(token_handler)
        self.user_info = user_info

    def initialize_metadata(self, output_dir, base_model_path, target_model_path) -> SimulatorMetadata:
        def create_metadata_with_status(status, error_message=None):
            metadata = SimulatorMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = SimulatorMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Simulator task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            metadata.base_model_path = Path(base_model_path).resolve().as_posix()
            metadata.target_model_path = Path(target_model_path).resolve().as_posix()
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def _upload_model_file(self, model_path: str) -> ResponseModelItem:
        # Get presigned_model_upload_url
        presigned_url_response = launcher_client_v2.simulator.presigned_model_upload_url(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=model_path,
        )

        # Upload model_file
        launcher_client_v2.simulator.upload_model_file(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=model_path,
            presigned_upload_url=presigned_url_response.data.presigned_upload_url,
        )

        # Validate model_file
        validate_model_response = launcher_client_v2.simulator.validate_model_file(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=model_path,
            ai_model_id=presigned_url_response.data.ai_model_id,
        )

        return validate_model_response

    def simulate_model(
        self,
        base_model_path: str,
        target_model_path: str,
        output_dir: str,
        dataset_path: Optional[str] = None,
    ) -> SimulatorMetadata:
        """Simulate a model to the specified framework.

        Args:
            base_model_path (str): The file path where the base model is located.
            target_model_path (str): The file path where the target model is located.
            output_dir (str): The local folder path to save the simulation result.
            dataset_path (str): The file path where the dataset is located.

        Raises:
            e: If an error occurs during the simulation.

        Returns:
            SimulatorMetadata: Simulator metadata.
        """

        netspresso_analytics.send_event(event_name="simulate_model_using_np")

        FileHandler.check_input_model_path(base_model_path)
        FileHandler.check_input_model_path(target_model_path)
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(
            output_dir=output_dir,
            base_model_path=base_model_path,
            target_model_path=target_model_path,
        )

        try:
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            base_model = self._upload_model_file(base_model_path)
            target_model = self._upload_model_file(target_model_path)

            # Start simulate task
            simulate_response = launcher_client_v2.simulator.start_task(
                access_token=self.token_handler.tokens.access_token,
                base_model_id=base_model.data.ai_model_id,
                target_model_id=target_model.data.ai_model_id,
                type=SimulateTaskType.OUTPUT,
                dataset_path=dataset_path,
            )

            metadata.base_model_info = base_model.data.to()
            metadata.target_model_info = target_model.data.to()
            metadata.simulate_task_info = simulate_response.data.to()
            metadata.status = Status.COMPLETED

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.MODEL_SIMULATE, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.MODEL_SIMULATE)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata
