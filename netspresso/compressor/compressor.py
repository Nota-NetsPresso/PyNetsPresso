from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.compressor import compressor_client
from netspresso.clients.compressor.schemas.compression import (
    AutoCompressionRequest,
    AvailableLayer,
    CompressionRequest,
    CreateCompressionRequest,
    GetAvailableLayersRequest,
    Options,
    RecommendationRequest,
    UploadDatasetRequest,
)
from netspresso.clients.compressor.schemas.model import UploadModelRequest
from netspresso.clients.launcher import launcher_client
from netspresso.compressor.core.compression import CompressionInfo
from netspresso.compressor.core.model import CompressedModel, Model, ModelCollection, ModelFactory
from netspresso.enums import CompressionMethod, Framework, Module, RecommendationMethod, ServiceCredit, Status, TaskType

from ..utils import FileHandler, check_credit_balance
from ..utils.metadata import MetadataHandler
from .utils.onnx import export_onnx


class Compressor:
    def __init__(self, token_handler: TokenHandler) -> None:
        """Initialize the Compressor."""

        self.token_handler = token_handler
        self.model_factory = ModelFactory()

    def upload_model(
        self,
        input_model_path: str,
        input_shapes: List[Dict[str, int]] = None,
        framework: Framework = Framework.PYTORCH,
    ) -> Model:
        """Upload a model for compression.

        Args:
            input_model_path (str): The file path where the model is located.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].
            framework (Framework): The framework of the model.

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            Model: Uploaded model object.
        """

        if input_shapes is None:
            input_shapes = []
        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            logger.info("Uploading Model...")

            model_name = Path(input_model_path).stem

            data = UploadModelRequest(
                model_name=model_name,
                framework=framework,
                file_path=input_model_path,
                input_layers=input_shapes,
            )
            model_info = compressor_client.upload_model(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            model = self.model_factory.create_model(model_info=model_info)

            logger.info(f"Upload model successfully. Model ID: {model.model_id}")

            return model

        except Exception as e:
            logger.error(f"Upload model failed. Error: {e}")
            raise e

    def get_model(self, model_id: str) -> Union[Model, CompressedModel]:
        """Get the model for a given model ID.

        Args:
            model_id (str): The ID of the model.

        Raises:
            e: If an error occurs while getting the model.

        Returns:
            Union[Model, CompressedModel]: The retrieved model. If the model is compressed,
            `CompressedModel` will be returned. Otherwise, `Model` will be returned.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Getting model...")
            model_info = compressor_client.get_model_info(
                model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            if model_info.status.is_compressed:
                model = self.model_factory.create_compressed_model(model_info=model_info)
            else:
                model = self.model_factory.create_model(model_info=model_info)
            logger.info("Get model successfully.")

            return model

        except Exception as e:
            logger.error(f"Get model failed. Error: {e}")
            raise e

    def download_model(self, model_id: str, local_path: str) -> None:
        """Download the model for a given model ID to the local path.

        Args:
            model_id (str): The ID of the model.
            local_path (str): The local path to save the downloaded model.

        Raises:
            e: If an error occurs while downloading the model.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Downloading model...")
            download_link = compressor_client.get_download_model_link(
                model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            request.urlretrieve(download_link.url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download model failed. Error: {e}")
            raise e

    def delete_model(self, model_id: str, recursive: bool = False) -> None:
        """Delete the model for a given model ID.

        Args:
            model_id (str): The ID of the model.
            recursive (bool, optional): Whether to also delete the compressed model for that model. Defaults to False.

        Raises:
            e: If an error occurs while deleting the model.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Deleting model...")
            children_models = compressor_client.get_children_models(
                model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            if len(children_models) != 0:
                if not recursive:
                    logger.warning(
                        "Deleting the model will also delete its compressed models. To proceed with the deletion, set the `recursive` parameter to True."
                    )
                else:
                    logger.info("The compressed model for that model will also be deleted.")
                    compressor_client.delete_model(
                        model_id=model_id,
                        access_token=self.token_handler.tokens.access_token,
                        verify_ssl=self.token_handler.verify_ssl,
                    )
                    logger.info("Delete model successfully.")
            else:
                logger.info("The model will be deleted.")
                compressor_client.delete_model(
                    model_id=model_id,
                    access_token=self.token_handler.tokens.access_token,
                    verify_ssl=self.token_handler.verify_ssl,
                )
                logger.info("Delete model successfully.")

        except Exception as e:
            logger.error(f"Delete model failed. Error: {e}")
            raise e

    def select_compression_method(
        self,
        model_id: str,
        compression_method: CompressionMethod,
        options: Options = Options(),
    ) -> CompressionInfo:
        """Select a compression method for a model.

        Args:
            model_id (str): The ID of the model.
            compression_method (CompressionMethod): The selected compression method.
            options(Options, optional): The options for pruning method.

        Raises:
            e: If an error occurs while selecting the compression method.

        Returns:
            CompressionInfo: The compression information for the selected compression method.
        """

        self.token_handler.validate_token()

        try:
            model = self.get_model(model_id)

            logger.info("Selecting compression method...")
            if model.framework == Framework.PYTORCH and compression_method == CompressionMethod.PR_NN:
                raise Exception("The Nuclear Norm is only supported in the TensorFlow-Keras framework.")

            data = GetAvailableLayersRequest(
                model_id=model.model_id,
                compression_method=compression_method,
                options=options,
            )
            response = compressor_client.get_available_layers(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            compression_info = CompressionInfo(
                original_model_id=model.model_id,
                compression_method=compression_method,
                options=options.dict(),
            )
            compression_info.set_available_layers(response.available_layers)
            logger.info("Select compression method successfully.")

            return compression_info

        except Exception as e:
            logger.error(f"Select compression method failed. Error: {e}")
            raise e

    def get_compression(self, compression_id: str) -> CompressionInfo:
        """Get information about a compression.

        Args:
            compression_id (str): The ID of the compression.

        Raises:
            e: If an error occurs while getting the compression information.

        Returns:
            CompressionInfo: The information about the compression.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Getting compression...")
            _compression_info = compressor_client.get_compression_info(
                compression_id=compression_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            compression_info = CompressionInfo(
                compressed_model_id=_compression_info.new_model_id,
                compression_id=_compression_info.compression_id,
                compression_method=_compression_info.compression_method,
            )
            compression_info.set_available_layers(_compression_info.available_layers)
            logger.info("Get compression successfully.")

            return compression_info

        except Exception as e:
            logger.error(f"Get compression failed. Error: {e}")
            raise e

    def __upload_dataset(self, model_id: str, dataset_path: str) -> None:
        """Upload a dataset for nuclear norm compression method.

        Args:
            model_id (str): The ID of the model.
            dataset_path (str): The file path where the dataset is located.

        Raises:
            e: If an error occurs while uploading the dataset.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Uploading dataset...")
            data = UploadDatasetRequest(model_id=model_id, file_path=dataset_path)
            compressor_client.upload_dataset(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info("Upload dataset successfully.")

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    def _get_available_devices(self, compressed_model, default_model_path: str):
        """Get the available devices for the compressed model.

        Args:
            compressed_model: The compressed model.
            default_model_path (str): Path to the default model file.

        Returns:
            str: The uploaded model after conversion.
        """

        if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
            export_onnx(default_model_path, compressed_model.input_shapes)
            converter_uploaded_model = launcher_client.upload_model(
                model_file_path=default_model_path.with_suffix(".onnx"),
                target_function=Module.CONVERT,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
        else:
            converter_uploaded_model = launcher_client.upload_model(
                model_file_path=default_model_path.with_suffix(".h5"),
                target_function=Module.CONVERT,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

        return converter_uploaded_model

    def compress_model(
        self,
        compression: CompressionInfo,
        output_dir: str,
        dataset_path: Optional[str] = None,
    ) -> Dict:
        """Compress a model using the provided compression information.

        Args:
            compression (CompressionInfo): The information about the compression.
            output_dir (str): The local path to save the compressed model.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while compressing the model.

        Returns:
            Dict: Source model and compressed model information.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Compressing model...")

            model_info = self.get_model(compression.original_model_id)

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=model_info.framework
            )
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.COMPRESS)

            current_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            check_credit_balance(
                user_credit=current_credit,
                service_credit=ServiceCredit.ADVANCED_COMPRESSION,
            )

            model_name = Path(output_dir).name

            data = CreateCompressionRequest(
                model_id=compression.original_model_id,
                model_name=model_name,
                compression_method=compression.compression_method,
                options=compression.options,
            )
            compression_info = compressor_client.create_compression(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )

            if dataset_path and compression.compression_method == CompressionMethod.PR_NN:
                self.__upload_dataset(model_id=compression.original_model_id, dataset_path=dataset_path)

            for available_layers in compression.available_layers:
                if available_layers.values != [""]:
                    available_layers.use = True

            all_layers_false = all(available_layer.values == [""] for available_layer in compression.available_layers)
            if all_layers_false:
                raise Exception(
                    "The available_layer.values all empty. please put in the available_layer.values to compress."
                )

            available_layers = [
                AvailableLayer(
                    name=layer.name,
                    values=layer.values,
                    channels=layer.channels,
                    use=layer.use,
                )
                for layer in compression.available_layers
            ]

            data = CompressionRequest(
                compression_id=compression_info.compression_id,
                compression_method=compression.compression_method,
                layers=available_layers,
                compressed_model_id=compression_info.new_model_id,
                options=compression.options,
            )
            compressor_client.compress_model(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )

            self.download_model(
                model_id=compression_info.new_model_id,
                local_path=default_model_path.with_suffix(extension),
            )
            compressed_model = self.get_model(model_id=compression_info.new_model_id)

            converter_uploaded_model = self._get_available_devices(compressed_model, default_model_path)

            logger.info(f"Compress model successfully. Compressed Model ID: {compressed_model.model_id}")
            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(
                f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            metadata.update_compressed_model_path(
                compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            )
            if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
                metadata.update_compressed_onnx_model_path(
                    compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
                )
            metadata.update_model_info(
                task=model_info.task,
                framework=model_info.framework,
                input_shapes=model_info.input_shapes,
            )
            metadata.update_compression_info(
                method=compression.compression_method,
                options=compression.options,
                layers=compression.available_layers,
            )
            metadata.update_results(model=model_info, compressed_model=compressed_model)
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_devices(converter_uploaded_model.available_devices)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return metadata.asdict()

        except Exception as e:
            logger.error(f"Compress model failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def recommendation_compression(
        self,
        compression_method: CompressionMethod,
        recommendation_method: RecommendationMethod,
        recommendation_ratio: float,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        options: Options = Options(),
        dataset_path: Optional[str] = None,
    ) -> Dict:
        """Compress a recommendation-based model using the given compression and recommendation methods.

        Args:
            compression_method (CompressionMethod): The selected compression method.
            recommendation_method (RecommendationMethod): The selected recommendation method.
            recommendation_ratio (float): The compression ratio recommended by the recommendation method.
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local path to save the compressed model.
            input_shapes (List[Dict[str, int]]): Input shapes of the model.
            framework (Framework, optional): The framework of the model.
            options(Options, optional): The options for pruning method.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while performing recommendation compression.

        Returns:
            Dict: Source model and compressed model information.
        """

        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            logger.info("Compressing recommendation-based model...")

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=framework
            )
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.COMPRESS)

            current_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            check_credit_balance(
                user_credit=current_credit,
                service_credit=ServiceCredit.ADVANCED_COMPRESSION,
            )

            if framework == Framework.PYTORCH and compression_method == CompressionMethod.PR_NN:
                raise Exception("The Nuclear Norm is only supported in the TensorFlow-Keras framework.")

            if compression_method in [CompressionMethod.PR_ID, CompressionMethod.FD_CP]:
                raise Exception(
                    f"The {compression_method} compression method you choose doesn't provide a recommendation."
                )

            if (
                compression_method
                in [
                    CompressionMethod.PR_L2,
                    CompressionMethod.PR_GM,
                    CompressionMethod.PR_NN,
                ]
                and recommendation_method != RecommendationMethod.SLAMP
            ):
                raise Exception(
                    f"The {compression_method} compression method is only available the SLAMP recommendation method."
                )

            if (
                compression_method in [CompressionMethod.FD_TK, CompressionMethod.FD_SVD]
                and recommendation_method != RecommendationMethod.VBMF
            ):
                raise Exception(
                    f"The {compression_method} compression method is only available the VBMF recommendation method."
                )

            model_name = Path(output_dir).name

            model = self.upload_model(
                framework=framework,
                input_model_path=input_model_path,
                input_shapes=input_shapes,
            )

            data = CreateCompressionRequest(
                model_id=model.model_id,
                model_name=model_name,
                compression_method=compression_method,
                options=options.dict(),
            )
            compression_info = compressor_client.create_compression(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )

            if dataset_path and compression_method == CompressionMethod.PR_NN:
                self.__upload_dataset(model_id=model.model_id, dataset_path=dataset_path)

            data = RecommendationRequest(
                model_id=model.model_id,
                compression_id=compression_info.compression_id,
                recommendation_method=recommendation_method,
                recommendation_ratio=recommendation_ratio,
                options=options.dict(),
            )
            logger.info("Compressing model...")
            recommendation_result = compressor_client.get_recommendation(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )

            for recommended_layer in recommendation_result.recommended_layers:
                for available_layer in compression_info.available_layers:
                    # Find the matching available_layer by name
                    if available_layer.name == recommended_layer.name:
                        available_layer.use = True
                        available_layer.values = recommended_layer.values

            data = CompressionRequest(
                compression_id=compression_info.compression_id,
                compression_method=compression_method,
                layers=compression_info.available_layers,
                compressed_model_id=compression_info.new_model_id,
                options=options.dict(),
            )
            compressor_client.compress_model(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )

            self.download_model(
                model_id=compression_info.new_model_id,
                local_path=default_model_path.with_suffix(extension),
            )
            compressed_model = self.get_model(model_id=compression_info.new_model_id)

            converter_uploaded_model = self._get_available_devices(compressed_model, default_model_path)

            logger.info(f"Recommendation compression successfully. Compressed Model ID: {compressed_model.model_id}")
            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(
                f"{ServiceCredit.ADVANCED_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            _compression_info = self.get_compression(compression_info.compression_id)
            metadata.update_compressed_model_path(
                compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            )
            if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
                metadata.update_compressed_onnx_model_path(
                    compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
                )
            metadata.update_model_info(task=model.task, framework=framework, input_shapes=input_shapes)
            metadata.update_compression_info(
                method=_compression_info.compression_method,
                ratio=recommendation_ratio,
                options=_compression_info.options,
                layers=_compression_info.available_layers,
            )
            metadata.update_results(model=model, compressed_model=compressed_model)
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_devices(converter_uploaded_model.available_devices)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return metadata.asdict()

        except Exception as e:
            logger.error(f"Recommendation compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def automatic_compression(
        self,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        compression_ratio: float = 0.5,
    ) -> Dict:
        """Compress a model automatically based on the given compression ratio.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local path to save the compressed model.
            input_shapes (List[Dict[str, int]]): Input shapes of the model.
            framework (Framework, optional): The framework of the model.
            compression_ratio (float, optional): The compression ratio for automatic compression. Defaults to 0.5.

        Raises:
            e: If an error occurs while performing automatic compression.

        Returns:
            Dict: Source model and compressed model information.
        """

        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            logger.info("Compressing automatic-based model...")

            output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=framework
            )
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.COMPRESS)

            current_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            check_credit_balance(
                user_credit=current_credit,
                service_credit=ServiceCredit.AUTOMATIC_COMPRESSION,
            )

            model_name = Path(output_dir).name

            model = self.upload_model(
                framework=framework,
                input_model_path=input_model_path,
                input_shapes=input_shapes,
            )

            compressed_model_name = f"{model_name}_automatic_{compression_ratio}"
            data = AutoCompressionRequest(
                model_id=model.model_id,
                model_name=compressed_model_name,
                recommendation_ratio=compression_ratio,
                save_path=output_dir,
            )
            logger.info("Compressing model...")
            model_info = compressor_client.auto_compression(
                data=data, access_token=self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            compression_info = self.get_compression(model_info.original_compression_id)

            self.download_model(
                model_id=model_info.model_id,
                local_path=default_model_path.with_suffix(extension),
            )
            compressed_model = self.model_factory.create_compressed_model(model_info=model_info)  # TODO: delete

            converter_uploaded_model = self._get_available_devices(compressed_model, default_model_path)

            logger.info(f"Automatic compression successfully. Compressed Model ID: {compressed_model.model_id}")
            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, verify_ssl=self.token_handler.verify_ssl
            )
            logger.info(
                f"{ServiceCredit.AUTOMATIC_COMPRESSION} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            metadata.update_compressed_model_path(
                compressed_model_path=default_model_path.with_suffix(extension).as_posix()
            )
            if compressed_model.framework in [Framework.PYTORCH, Framework.ONNX]:
                metadata.update_compressed_onnx_model_path(
                    compressed_onnx_model_path=default_model_path.with_suffix(".onnx").as_posix()
                )
            metadata.update_model_info(task=model.task, framework=framework, input_shapes=input_shapes)
            metadata.update_compression_info(
                method=compression_info.compression_method,
                ratio=compression_ratio,
                options=compression_info.options,
                layers=compression_info.available_layers,
            )
            metadata.update_results(model=model, compressed_model=compressed_model)
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_devices(converter_uploaded_model.available_devices)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            return metadata.asdict()

        except Exception as e:
            logger.error(f"Automatic compression failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
