import time
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from netspresso.analytics import netspresso_analytics
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.benchmark.response_body import BenchmarkTask
from netspresso.enums import Status, TaskStatusForDisplay
from netspresso.enums.credit import ServiceTask
from netspresso.enums.device import DeviceName, HardwareType, SoftwareVersion
from netspresso.enums.model import DataType
from netspresso.metadata.profiler import ProfilerMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class Profiler(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse) -> None:
        """Initialize the Profiler."""

        super().__init__(token_handler)
        self.user_info = user_info

    def get_data_type(self, input_model_dir):
        metadata_path = input_model_dir / "metadata.json"
        try:
            if metadata_path.exists():
                metadata = MetadataHandler.load_json(metadata_path)
                convert_task_info = metadata.get("convert_task_info", {})
                return convert_task_info.get("data_type", DataType.FP32)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"Error loading metadata: {e}")

        return DataType.FP32

    def initialize_metadata(self, input_model_path: str):
        def create_metadata_with_status(status, error_message=None):
            metadata = ProfilerMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = ProfilerMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Profile task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            # Load existing metadata if available
            metadatas = []
            output_dir = Path(input_model_path).parent
            file_path = output_dir / "profile.json"
            if FileHandler.check_exists(file_path):
                metadatas = MetadataHandler.load_json(file_path)

            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            metadatas.append(metadata)
            MetadataHandler.save_benchmark_result(data=metadatas, folder_path=output_dir)

        return metadatas

    def profile_model(
        self,
        input_model_path: str,
        target_device_name: DeviceName,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        target_hardware_type: Optional[Union[str, HardwareType]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> ProfilerMetadata:
        """Profile the specified model on the specified device.

        Args:
            input_model_path (str): The file path where the model is located.
            target_device_name (DeviceName): Target device name.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version. Required if target_device_name is one of the Jetson devices.
            target_hardware_type (Union[str, HardwareType], optional): Hardware type. Acceleration options for processing the model inference.
            wait_until_done (bool): If True, wait for the profile result before returning the function.
                                If False, request the profile and return the function immediately.

        Raises:
            e: If an error occurs during the profiling of the model.

        Returns:
            ProfilerMetadata: Profiler metadata.
        """

        netspresso_analytics.send_event(
            event_name="profile_model_using_np",
            event_params={
                "target_device_name": target_device_name,
                "target_software_version": target_software_version or "",
                "target_hardware_type": target_hardware_type or "",
            },
        )

        FileHandler.check_input_model_path(input_model_path)
        metadatas = self.initialize_metadata(input_model_path=input_model_path)

        try:
            metadata: ProfilerMetadata = metadatas[-1]
            output_dir = Path(input_model_path).parent
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_PROFILE)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.benchmarker.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.benchmarker.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.benchmarker.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start benchmark task
            benchmark_response = launcher_client_v2.benchmarker.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                data_type=validate_model_response.data.detail.data_type,
                target_device_name=target_device_name,
                hardware_type=target_hardware_type,
                input_layer=validate_model_response.data.detail.input_layers[0],
                software_version=target_software_version,
            )

            metadata.profile_task_info = benchmark_response.data.to()
            metadata.profile_task_info.data_type = self.get_data_type(output_dir)
            MetadataHandler.save_benchmark_result(data=metadatas, folder_path=output_dir)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    benchmark_response = launcher_client_v2.benchmarker.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=benchmark_response.data.benchmark_task_id,
                    )
                    if benchmark_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if benchmark_response.data.status == TaskStatusForDisplay.FINISHED:
                self.print_remaining_credit(service_task=ServiceTask.MODEL_PROFILE)
                metadata.status = Status.COMPLETED
                metadata.profile_result = benchmark_response.data.benchmark_result.to(
                    file_size=validate_model_response.data.file_size_in_mb
                )
                logger.info("Profiling task was completed successfully.")
            else:
                metadata = self.handle_error(metadata, ServiceTask.MODEL_PROFILE, benchmark_response.data.error_log)

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.MODEL_PROFILE, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.MODEL_PROFILE)
        finally:
            metadatas[-1] = metadata
            MetadataHandler.save_benchmark_result(data=metadatas, folder_path=output_dir)

        return metadata

    def get_profile_task(self, profile_task_id: str) -> BenchmarkTask:
        """Get information about the specified profile task using the profile task UUID.

        Args:
            profile_task_id (str): Profile task UUID of the profile task.

        Raises:
            e: If an error occurs while retrieving information about the profile task.

        Returns:
            BenchmarkTask: Model profile task object.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=profile_task_id,
        )
        return response.data

    def cancel_profile_task(self, profile_task_id: str) -> BenchmarkTask:
        """Cancel the profile task with given profile task uuid.

        Args:
            profile_task_id (str): Profile task UUID of the profile task.

        Raises:
            e: If an error occurs during the task cancel.

        Returns:
            BenchmarkTask: Model profile task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=profile_task_id,
        )
        return response.data
