import time
from pathlib import Path
from typing import List, Optional
from urllib import request

from loguru import logger

from netspresso.analytics import netspresso_analytics
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.graph_optimize.response_body import GraphOptimizeTask
from netspresso.enums import ServiceTask, Status, TaskStatusForDisplay
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler
from netspresso.metadata.graph_optimizer import GraphOptimizerMetadata
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler
from netspresso.utils.onnx import update_onnx_input_batch_size_as_1


class GraphOptimizer(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the GraphOptimizer."""

        super().__init__(token_handler)
        self.user_info = user_info

    def _download_optimized_model(self, graph_optimize_task: GraphOptimizeTask, local_path: str) -> None:
        """Download the optimized model with given graph optimize task.

        Args:
            graph_optimize_task (GraphOptimizeTask): Graph optimize task.

        Raises:
            e: If an error occurs while getting the graph optimize task information.
        """

        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.graph_optimizer.download_model_file(
                graph_optimize_task_id=graph_optimize_task.graph_optimize_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download optimized model failed. Error: {e}")
            raise e

    def initialize_metadata(self, output_dir, input_model_path, pattern_handlers) -> GraphOptimizerMetadata:
        def create_metadata_with_status(status, error_message=None):
            metadata = GraphOptimizerMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = GraphOptimizerMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Graph Optimize task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            metadata.graph_optimize_task_info.pattern_handlers = pattern_handlers
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def optimize_model(
        self,
        input_model_path: str,
        output_dir: str,
        pattern_handlers: Optional[List[GraphOptimizePatternHandler]] = GraphOptimizePatternHandler.get_all(),
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> GraphOptimizerMetadata:
        """Optimize a model to the specified framework.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the optimized model.
            pattern_handlers (List[GraphOptimizePatternHandler]): The pattern handlers to optimize the model.
            wait_until_done (bool): If True, wait for the graph optimize result before returning the function.
                                If False, request the graph optimize and return  the function immediately.

        Raises:
            e: If an error occurs during the graph optimize.

        Returns:
            GraphOptimizerMetadata: Graph optimize metadata.
        """

        netspresso_analytics.send_event(event_name="optimize_model_using_np")

        self.token_handler.validate_token()

        FileHandler.check_input_model_path(input_model_path)
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(
            output_dir=output_dir,
            input_model_path=input_model_path,
            pattern_handlers=pattern_handlers,
        )

        try:
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            # Check if the model is supported
            batch_size_1_model_path = Path(output_dir) / f"{Path(input_model_path).stem}_batch_size_1.onnx"
            input_model_path = update_onnx_input_batch_size_as_1(input_model_path, batch_size_1_model_path.as_posix())

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.graph_optimizer.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.graph_optimizer.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.graph_optimizer.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start graph optimize task
            graph_optimize_response = launcher_client_v2.graph_optimizer.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                pattern_handlers=pattern_handlers,
            )

            metadata.model_info = validate_model_response.data.to()
            metadata.graph_optimize_task_info = graph_optimize_response.data.to(validate_model_response.data.uploaded_file_name)
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    graph_optimize_response = launcher_client_v2.graph_optimizer.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=graph_optimize_response.data.graph_optimize_task_id,
                    )
                    if graph_optimize_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if graph_optimize_response.data.status == TaskStatusForDisplay.FINISHED:
                default_model_path = FileHandler.get_default_model_path(folder_path=output_dir)
                extension = ".onnx"
                self._download_optimized_model(
                    graph_optimize_task=graph_optimize_response.data,
                    local_path=str(default_model_path.with_suffix(extension)),
                )
                metadata.status = Status.COMPLETED
                metadata.optimized_model_path = default_model_path.with_suffix(extension).as_posix()
                metadata.graph_optimize_task_info = graph_optimize_response.data.to(validate_model_response.data.uploaded_file_name)
                logger.info("Graph Optimize task was completed successfully.")
            else:
                metadata = self.handle_error(metadata, ServiceTask.MODEL_GRAPH_OPTIMIZE, graph_optimize_response.data.error_log)

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.MODEL_GRAPH_OPTIMIZE, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.MODEL_GRAPH_OPTIMIZE)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def get_graph_optimize_task(self, graph_optimize_task_id: str) -> GraphOptimizeTask:
        """Get the graph optimize task information with given graph optimize task uuid.

        Args:
            graph_optimize_task_id (str): Graph optimize task UUID of the graph optimize task.

        Raises:
            e: If an error occurs during the graph optimize task.

        Returns:
            GraphOptimizeTask: Graph optimize task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.graph_optimizer.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=graph_optimize_task_id,
        )
        return response.data
