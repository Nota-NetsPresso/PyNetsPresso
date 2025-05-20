import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib import request

from loguru import logger

from netspresso.analytics import netspresso_analytics
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.task.quantize.request_body import (
    AutomaticQuantizeOption,
    CustomQuantizeOption,
    PlainQuantizationOption,
    RecommendationOption,
)
from netspresso.clients.launcher.v2.schemas.task.quantize.response_body import QuantizeTask
from netspresso.enums import (
    QuantizationMode,
    QuantizationPrecision,
    ServiceTask,
    SimilarityMetric,
    Status,
    TaskStatusForDisplay,
)
from netspresso.metadata.quantizer import QuantizerMetadata
from netspresso.quantizer.schema import PrecisionByLayer, PrecisionByOperator, RecommendationPrecisions
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class Quantizer(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Quantizer."""

        super().__init__(token_handler)
        self.user_info = user_info

    def initialize_metadata(
        self,
        output_dir,
        input_model_path,
        threshold,
        quantization_mode,
        metric,
        weight_precision,
        activation_precision,
    ):
        def create_metadata_with_status(status, error_message=None):
            metadata = QuantizerMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = QuantizerMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Quantization task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            metadata.quantize_info.threshold = threshold
            metadata.quantize_info.quantization_mode = quantization_mode
            metadata.quantize_info.metric = metric
            metadata.quantize_info.weight_precision = weight_precision
            metadata.quantize_info.activation_precision = activation_precision
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def _download_quantized_model(
        self, quantize_task: QuantizeTask, output_dir: str, metadata: QuantizerMetadata
    ) -> None:
        """Download the quantizeed model with given quantization task or quantization task uuid.

        Args:
            quantization_task (QuantizeTask): Launcher Model Object or the uuid of the quantization task.

        Raises:
            e: If an error occurs while getting the quantization task information.
        """

        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.quantizer.download_model_file(
                quantize_task_uuid=quantize_task.quantize_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            default_model_path = FileHandler.get_default_model_path(folder_path=output_dir)
            download_model_path = default_model_path.with_suffix(".zip").as_posix()

            request.urlretrieve(download_url, download_model_path)
            logger.info(f"Model downloaded at {Path(download_model_path)}")

            metadata.quantized_model_path = download_model_path
            FileHandler.unzip(zip_file_path=download_model_path, target_path=output_dir)
            FileHandler.remove_file(file_path=download_model_path)

            old_file_path = Path(output_dir) / "quantized_qdq.onnx"
            quantized_model_path = default_model_path.with_suffix(".onnx").as_posix()
            metadata.quantized_model_path = old_file_path
            FileHandler.rename_file(old_file_path=old_file_path, new_file_path=quantized_model_path)

            compare_result = FileHandler.load_json(file_path=output_dir / "snr_compare_result.json")

            self.print_remaining_credit(service_task=ServiceTask.MODEL_QUANTIZE)

            metadata.status = Status.COMPLETED
            metadata.quantized_model_path = quantized_model_path
            metadata.compare_result = compare_result

            return metadata

        except Exception as e:
            logger.error(f"Download quantized model failed. Error: {e}")
            raise e

    def _download_recommendation_result(
        self, quantize_task: QuantizeTask, output_dir: str, metadata: QuantizerMetadata
    ) -> None:
        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.quantizer.download_model_file(
                quantize_task_uuid=quantize_task.quantize_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            download_path = (Path(output_dir) / "custom_quantization_suggestion.json").resolve().as_posix()

            request.urlretrieve(download_url, download_path)
            logger.info(f"Model downloaded at {Path(download_path)}")

            self.print_remaining_credit(service_task=ServiceTask.MODEL_QUANTIZE)

            metadata.status = Status.COMPLETED
            metadata.recommendation_result_path = download_path

            return metadata

        except Exception as e:
            logger.error(f"Download quantized model failed. Error: {e}")
            raise e

    def _upload_model(self, input_model_path: str):
        # Get presigned_model_upload_url
        presigned_url_response = launcher_client_v2.quantizer.presigned_model_upload_url(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=input_model_path,
        )

        # Upload model_file
        launcher_client_v2.quantizer.upload_model_file(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=input_model_path,
            presigned_upload_url=presigned_url_response.data.presigned_upload_url,
        )

        # Validate model_file
        validate_model_response = launcher_client_v2.quantizer.validate_model_file(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=input_model_path,
            ai_model_id=presigned_url_response.data.ai_model_id,
        )

        return validate_model_response

    def _quantize_model(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        quantization_mode: QuantizationMode,
        quantization_options: Union[PlainQuantizationOption, CustomQuantizeOption, AutomaticQuantizeOption],
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        """Quantize a model to the specified framework.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            quantization_mode (QuantizationMode): Quantization mode
            input_layers (List[InputShape], optional): Target input shape for quantization (e.g., dynamic batch to static batch).
            wait_until_done (bool): If True, wait for the quantization result before returning the function.
                                If False, request the quantization and return  the function immediately.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantize metadata.
        """

        FileHandler.check_input_model_path(input_model_path)
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(
            output_dir=output_dir,
            input_model_path=input_model_path,
            threshold=0,
            quantization_mode=quantization_mode,
            metric=quantization_options.metric,
            weight_precision=QuantizationPrecision.INT8,
            activation_precision=QuantizationPrecision.INT8,
        )

        try:
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_QUANTIZE)

            uploaded_model_response = self._upload_model(input_model_path=input_model_path)

            # Start quantize task
            input_layers = input_layers if input_layers else uploaded_model_response.data.detail.input_layers
            quantize_response = launcher_client_v2.quantizer.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=uploaded_model_response.data.ai_model_id,
                quantization_mode=quantization_mode,
                quantization_options=quantization_options,
                input_layers=input_layers,
                dataset_path=dataset_path,
            )

            metadata.model_info = uploaded_model_response.data.to()
            metadata.quantize_info = quantize_response.data.to(uploaded_model_response.data.uploaded_file_name)
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    quantize_response = launcher_client_v2.quantizer.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=quantize_response.data.quantize_task_id,
                    )
                    if quantize_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if quantize_response.data.status == TaskStatusForDisplay.FINISHED:
                if quantize_response.data.quantization_mode in [
                    "plain_quantization",
                    "custom_quantization",
                    "automatic_quantization",
                ]:
                    metadata = self._download_quantized_model(quantize_response.data, output_dir, metadata)
                elif quantize_response.data.quantization_mode in [QuantizationMode.RECOMMEND_QUANTIZATION]:
                    metadata = self._download_recommendation_result(quantize_response.data, output_dir, metadata)

            else:
                metadata = self.handle_error(metadata, ServiceTask.MODEL_QUANTIZE, quantize_response.data.error_log)

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.MODEL_QUANTIZE, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.MODEL_QUANTIZE)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def uniform_precision_quantization(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        metric: SimilarityMetric = SimilarityMetric.SNR,
        weight_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        activation_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ):
        """Apply uniform precision quantization to a model, specifying precision for weight & activation.

        This method quantizes all layers in the model uniformly based on the specified precision levels for weights and activations.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            metric (SimilarityMetric): Quantization quality metrics.
            weight_precision (QuantizationPrecision): Weight precision
            activation_precision (QuantizationPrecision): Activation precision
            input_layers (List[InputShape], optional): Target input shape for quantization (e.g., dynamic batch to static batch).
            wait_until_done (bool): If True, wait for the quantization result before returning the function.
                                If False, request the quantization and return  the function immediately.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantize metadata.
        """
        netspresso_analytics.send_event(
            event_name="uniform_precision_quantization_using_np",
            event_params={
                "weight_precision": weight_precision,
                "activation_precision": activation_precision,
                "metric": metric,
            },
        )

        quantization_options = PlainQuantizationOption(
            metric=metric,
            weight_precision=weight_precision,
            activation_precision=activation_precision,
        )

        metadata = self._quantize_model(
            input_model_path=input_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            quantization_mode=QuantizationMode.UNIFORM_PRECISION_QUANTIZATION,
            quantization_options=quantization_options,
            input_layers=input_layers,
            wait_until_done=wait_until_done,
            sleep_interval=sleep_interval,
        )

        logger.info("Plain quantization task was completed successfully.")

        return metadata

    def automatic_quantization(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        weight_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        activation_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        metric: SimilarityMetric = SimilarityMetric.SNR,
        threshold: Union[float, int] = 0,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        """Apply automatic quantization to a model, specifying precision for weight & activation.

        This method quantizes layers in the model based on the specified precision levels for weights and activations, while evaluating
        the quality of quantization using the defined metric. Only layers that meet the specified quality `threshold` are quantized;
        layers that do not meet this threshold remain unquantized to preserve model accuracy.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            weight_precision (QuantizationPrecision): Weight precision
            activation_precision (QuantizationPrecision): Activation precision
            metric (SimilarityMetric): Quantization quality metrics.
            threshold (Union[float, int]): Quality threshold for quantization. Layers that do not meet this threshold based on the metric are not quantized.
            input_layers (List[InputShape], optional): Target input shape for quantization (e.g., dynamic batch to static batch).
            wait_until_done (bool): If True, wait for the quantization result before returning the function.
                                If False, request the quantization and return  the function immediately.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantize metadata.
        """
        netspresso_analytics.send_event(
            event_name="automatic_quantization_using_np",
            event_params={
                "weight_precision": weight_precision,
                "activation_precision": activation_precision,
                "metric": metric,
                "threshold": threshold,
            },
        )

        quantization_options = AutomaticQuantizeOption(
            metric=metric,
            threshold=threshold,
            weight_precision=weight_precision,
            activation_precision=activation_precision,
        )

        metadata = self._quantize_model(
            input_model_path=input_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            quantization_mode=QuantizationMode.AUTOMATIC_QUANTIZATION,
            quantization_options=quantization_options,
            input_layers=input_layers,
            wait_until_done=wait_until_done,
            sleep_interval=sleep_interval,
        )

        logger.info("Automatic quantization task was completed successfully.")

        return metadata

    def _custom_quantization(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        custom_quantization_dictionary: Dict,
        metric: SimilarityMetric = SimilarityMetric.SNR,
        weight_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        activation_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        quantization_options = CustomQuantizeOption(
            metric=metric,
            custom_precision=custom_quantization_dictionary,
            weight_precision=weight_precision,
            activation_precision=activation_precision,
        )

        metadata = self._quantize_model(
            input_model_path=input_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            quantization_mode=QuantizationMode.CUSTOM_PRECISION_QUANTIZATION,
            quantization_options=quantization_options,
            input_layers=input_layers,
            wait_until_done=wait_until_done,
            sleep_interval=sleep_interval,
        )

        return metadata

    def custom_precision_quantization_by_layer_name(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        precision_by_layer_name: List[PrecisionByLayer],
        default_weight_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        default_activation_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        metric: SimilarityMetric = SimilarityMetric.SNR,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        """
        Apply custom precision quantization to a model, specifying precision for each layer name.

        This function allows precise control over the quantization process by enabling the user to
        specify quantization precision (e.g., INT8, FP16) for each named layer within the model.
        The `precision_by_layer_name` parameter provides a list where each item details the target
        precision for a specific layer name, enabling customized quantization that can enhance
        model performance or compatibility.

        Users can target specific layers to be quantized to lower precision for optimized model
        size and performance while keeping critical layers at higher precision for accuracy.
        Layers not explicitly listed in `precision_by_layer_name` will use
        `default_weight_precision` and `default_activation_precision`.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            precision_by_layer_name (List[PrecisionByLayer]):
                List of `PrecisionByLayer` objects that specify the desired precision for each
                layer name in the model. Each entry includes: `name` (str): The layer name (e.g., /backbone/conv_first/block/act/Mul_output_0).
                `precision` (QuantizationPrecision): The quantization precision level.
            default_weight_precision (QuantizationPrecision): Weight precision.
            default_activation_precision (QuantizationPrecision): Activation precision.
            metric (SimilarityMetric): Quantization quality metrics.
            input_layers (List[InputShape], optional):
                Target input shape for quantization (e.g., dynamic batch to static batch).
            wait_until_done (bool): If True, wait for the quantization result before returning
                the function. If False, request the quantization and return immediately.
            sleep_interval (int): Interval in seconds between checks when `wait_until_done` is True.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantization metadata containing status, paths, etc.
        """
        netspresso_analytics.send_event(
            event_name="custom_precision_quantization_by_layer_name_using_np",
            event_params={
                "weight_precision": default_weight_precision,
                "activation_precision": default_activation_precision,
                "metric": metric,
            },
        )

        layers = {layer.name: layer.precision for layer in precision_by_layer_name}

        custom_quantization_dictionary = {"layers": layers, "operators": {}}

        metadata = self._custom_quantization(
            input_model_path=input_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            custom_quantization_dictionary=custom_quantization_dictionary,
            metric=metric,
            weight_precision=default_weight_precision,
            activation_precision=default_activation_precision,
            input_layers=input_layers,
            wait_until_done=wait_until_done,
            sleep_interval=sleep_interval,
        )

        logger.info("Custom quantization by layer name task was completed successfully.")

        return metadata

    def custom_precision_quantization_by_operator_type(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        precision_by_operator_type: List[PrecisionByOperator],
        default_weight_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        default_activation_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        metric: SimilarityMetric = SimilarityMetric.SNR,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        """
        Apply custom quantization to a model, specifying precision for each operator type.

        This function allows for highly customizable quantization by enabling the user to specify
        the quantization precision (e.g., INT8, FP16) for each operator type within a model. The
        `precision_by_operator_type` parameter is a list of mappings where each entry indicates
        the quantization precision for a specific operator type, such as convolution (Conv),
        matrix multiplication (MatMul), etc.

        Using `precision_by_operator_type`, users can selectively fine-tune the quantization
        strategy for different operators within the model, based on performance requirements
        or hardware capabilities. Operators not explicitly specified in
        `precision_by_operator_type` will fall back to `default_weight_precision` and
        `default_activation_precision`.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            precision_by_operator_type (List[PrecisionByOperator]):
                List of `PrecisionByOperator` objects that specify the desired precision for each
                operator type in the model. Each entry includes: `type` (str): The operator type (e.g., Conv, MatMul). `precision` (QuantizationPrecision): The quantization precision level.
            default_weight_precision (QuantizationPrecision): Weight precision.
            default_activation_precision (QuantizationPrecision): Activation precision.
            metric (SimilarityMetric): Quantization quality metrics.
            input_layers (List[InputShape], optional):
                Target input shape for quantization (e.g., dynamic batch to static batch).
            wait_until_done (bool): If True, wait for the quantization result before returning
                the function. If False, request the quantization and return immediately.
            sleep_interval (int): Interval in seconds between checks when `wait_until_done` is True.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantization metadata containing status, paths, etc.
        """
        netspresso_analytics.send_event(
            event_name="custom_precision_quantization_by_operator_type_using_np",
            event_params={
                "weight_precision": default_weight_precision,
                "activation_precision": default_activation_precision,
                "metric": metric,
            },
        )

        operators = {layer.type: layer.precision for layer in precision_by_operator_type}
        custom_quantization_dictionary = {"layers": {}, "operators": operators}

        metadata = self._custom_quantization(
            input_model_path=input_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            custom_quantization_dictionary=custom_quantization_dictionary,
            metric=metric,
            weight_precision=default_weight_precision,
            activation_precision=default_activation_precision,
            input_layers=input_layers,
            wait_until_done=wait_until_done,
            sleep_interval=sleep_interval,
        )

        logger.info("Custom quantization by operator type task was completed successfully.")

        return metadata

    def get_recommendation_precision(
        self,
        input_model_path: str,
        output_dir: str,
        dataset_path: Optional[str],
        weight_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        activation_precision: QuantizationPrecision = QuantizationPrecision.INT8,
        metric: SimilarityMetric = SimilarityMetric.SNR,
        threshold: Union[float, int] = 0,
        input_layers: List[Dict[str, int]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
    ) -> QuantizerMetadata:
        """Get recommended precision for a model based on a specified quality threshold.

        This function analyzes each layer of the given model and recommends precision settings
        for layers that do not meet the specified threshold, helping to balance quantization
        quality and performance.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the quantized model.
            dataset_path (str): Path to the dataset. Useful for certain quantizations.
            weight_precision (QuantizationPrecision): Target precision for weights.
            activation_precision (QuantizationPrecision): Target precision for activations.
            metric (SimilarityMetric): Metric used to evaluate quantization quality.
            threshold (Union[float, int]): Quality threshold; layers below this threshold will
                            receive precision recommendations.
            input_layers (List[Dict[str, int]], optional): Specifications for input shapes
                            (e.g., to convert from dynamic to static batch size).
            wait_until_done (bool): If True, waits for the quantization process to finish
                            before returning. If False, starts the process and returns immediately.
            sleep_interval (int): Interval, in seconds, between checks when `wait_until_done`
                            is True.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizerMetadata: Quantize metadata.

        """
        netspresso_analytics.send_event(
            event_name="get_recommendation_precision_using_np",
            event_params={
                "weight_precision": weight_precision,
                "activation_precision": activation_precision,
                "metric": metric,
                "threshold": threshold,
            },
        )

        quantization_options = RecommendationOption(
            metric=metric,
            threshold=threshold,
            weight_precision=weight_precision,
            activation_precision=activation_precision,
        )

        metadata = self._quantize_model(
            input_model_path=input_model_path,
            output_dir=output_dir,
            dataset_path=dataset_path,
            quantization_mode=QuantizationMode.RECOMMEND_QUANTIZATION,
            quantization_options=quantization_options,
            input_layers=input_layers,
            wait_until_done=wait_until_done,
            sleep_interval=sleep_interval,
        )

        logger.info("Get recommendation precision task was completed successfully.")

        return metadata

    def load_recommendation_precision_result(self, file_path: str):
        recommendation_result = FileHandler.load_json(file_path=file_path)
        layers = recommendation_result["layers"]
        operators = recommendation_result["operators"]

        precision_by_layer_name = [
            PrecisionByLayer(
                name=layer["name"],
                precision=layer["recommendation"]["precision"][0],
            )
            for layer in layers
        ]
        precision_by_operator_type = [
            PrecisionByOperator(
                type=operator["type"],
                precision=operator["recommendation"]["precision"][0],
            )
            for operator in operators
        ]

        recommendation_precisions = RecommendationPrecisions(
            layers=precision_by_layer_name,
            operators=precision_by_operator_type,
        )

        return recommendation_precisions

    def get_quantization_task(self, quantization_task_id: str) -> QuantizeTask:
        """Get the quantization task information with given quantization task uuid.

        Args:
            quantization_task_id (str): Quantize task UUID of the quantize task.

        Raises:
            e: If an error occurs during the model quantization.

        Returns:
            QuantizeTask: Model quantization task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.quantizer.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=quantization_task_id,
        )
        return response.data

    def cancel_quantization_task(self, quantization_task_id: str) -> QuantizeTask:
        """Cancel the quantization task with given quantization task uuid.

        Args:
            quantization_task_id (str): Quantize task UUID of the quantize task.

        Raises:
            e: If an error occurs during the task cancel.

        Returns:
            QuantizeTask: Model quantization task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.quantizer.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=quantization_task_id,
        )
        return response.data
