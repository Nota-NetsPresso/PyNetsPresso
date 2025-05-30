import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request

from loguru import logger

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.compressor import compressor_client_v2
from netspresso.clients.compressor.v2.schemas import (
    ModelBase,
    Options,
    RecommendationOptions,
    RequestAutomaticCompressionParams,
    RequestAvailableLayers,
    RequestCreateCompression,
    RequestCreateModel,
    RequestCreateRecommendation,
    RequestUpdateCompression,
    RequestUploadModel,
    RequestValidateModel,
    ResponseCompression,
    ResponseSelectMethod,
    UploadFile,
)
from netspresso.clients.launcher import launcher_client_v2
from netspresso.compressor.utils.onnx import export_onnx
from netspresso.enums import CompressionMethod, Framework, RecommendationMethod, ServiceTask, Status
from netspresso.enums.project import SubFolder
from netspresso.exceptions.compressor import FailedUploadModelException
from netspresso.metadata.compressor import CompressorMetadata
from netspresso.utils import FileHandler
from netspresso.utils.db.models.compression import CompressionModelResult, CompressionTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.models.training import TrainingTask
from netspresso.utils.db.repositories.compression import (
    compression_model_result_repository,
    compression_task_repository,
)
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository
from netspresso.utils.db.session import get_db_session
from netspresso.utils.metadata import MetadataHandler

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"

class CompressorV2(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse) -> None:
        """Initialize the Compressor."""
        super().__init__(token_handler)
        self.user_info = user_info

    def _update_metadata_for_trainer(self, metadata: CompressorMetadata, input_model_path: str):
        if (Path(input_model_path).parent / "metadata.json").exists():
            trained_data = FileHandler.load_json(Path(input_model_path).parent / "metadata.json")
            metadata.update_model_info_for_trainer(
                task=trained_data["model_info"]["task"],
                model=trained_data["model_info"]["model"],
                dataset=trained_data["model_info"]["dataset"],
            )
            metadata.update_training_info(
                epochs=trained_data["training_info"]["epochs"],
                batch_size=trained_data["training_info"]["batch_size"],
                learning_rate=trained_data["training_info"]["learning_rate"],
                optimizer=trained_data["training_info"]["optimizer"],
            )
            metadata.update_is_retrainable(is_retrainable=True)
            metadata.update_training_result(training_result=trained_data["training_result"])

        return metadata

    def initialize_metadata(self, output_dir, input_model_path, compression_method, ratio, framework, input_shapes):
        def create_metadata_with_status(status, error_message=None):
            metadata = CompressorMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = CompressorMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Compression task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            metadata.input_model_path = Path(input_model_path).resolve().as_posix()
            metadata.compression_info.method = compression_method
            metadata.compression_info.ratio = ratio
            metadata.model_info.framework = framework
            metadata.model_info.input_shapes = input_shapes
            metadata = self._update_metadata_for_trainer(metadata, input_model_path)
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def _get_available_options(self, compressed_model_info, default_model_path: str):
        if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
            export_onnx(default_model_path, compressed_model_info.detail.input_layers)
            options_response = launcher_client_v2.converter.read_framework_options(
                access_token=self.token_handler.tokens.access_token,
                framework=Framework.ONNX,
            )
        else:
            options_response = launcher_client_v2.converter.read_framework_options(
                access_token=self.token_handler.tokens.access_token,
                framework=Framework.TENSORFLOW_KERAS,
            )

        available_options = options_response.data

        # TODO: Will be removed when we support DLC in the future
        available_options = [
            available_option for available_option in available_options if available_option.framework != "dlc"
        ]

        return available_options

    def _postprocess_metadata(
        self,
        metadata: CompressorMetadata,
        model_info: ModelBase,
        compression_info: ResponseCompression,
        default_model_path: str,
        extension: str,
    ):
        compressed_model_info = self.get_model(model_id=compression_info.input_model_id)
        available_options = self._get_available_options(compressed_model_info, default_model_path)

        if compressed_model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
            metadata.update_compressed_onnx_model_path(default_model_path.with_suffix(".onnx").as_posix())
        metadata.compression_info.layers = compression_info.available_layers
        metadata.compression_info.options = compression_info.options
        metadata.update_compressed_model_path(default_model_path.with_suffix(extension).as_posix())
        metadata.update_results(model=model_info, compressed_model=compressed_model_info)
        metadata.update_status(status=Status.COMPLETED)
        metadata.update_available_options(available_options)

        return metadata

    def finalize_compression_process(
        self,
        metadata: CompressorMetadata,
        model_info: ModelBase,
        compression_info: ResponseCompression,
        output_dir: str,
    ):
        default_model_path = FileHandler.get_default_model_path(folder_path=output_dir)
        extension = FileHandler.get_extension(framework=model_info.detail.framework)
        self.download_model(compression_info.input_model_id, default_model_path.with_suffix(extension))
        metadata = self._postprocess_metadata(metadata, model_info, compression_info, default_model_path, extension)

        return metadata

    def upload_model(
        self,
        input_model_path: str,
        input_shapes: List[Dict[str, int]] = None,
        framework: Framework = Framework.PYTORCH,
    ) -> ModelBase:
        """Upload a model for compression.

        Args:
            input_model_path (str): The file path where the model is located.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].
            framework (Framework): The framework of the model.

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            ModelBase: Uploaded model object.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Uploading Model...")

            FileHandler.check_input_model_path(input_model_path)

            object_name = Path(input_model_path).name

            create_model_request = RequestCreateModel(object_name=object_name)
            create_model_response = compressor_client_v2.create_model(
                request_data=create_model_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            file_content = FileHandler.read_file_bytes(file_path=input_model_path)
            upload_model_request = RequestUploadModel(url=create_model_response.data.presigned_url)
            file = UploadFile(file_name=object_name, file_content=file_content)
            upload_model_response = compressor_client_v2.upload_model(
                request_data=upload_model_request,
                file=file,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            if not upload_model_response:
                # TODO: Confirm upload success
                raise FailedUploadModelException()

            validate_model_request = RequestValidateModel(framework=framework, input_layers=input_shapes)
            validate_model_response = compressor_client_v2.validate_model(
                ai_model_id=create_model_response.data.ai_model_id,
                request_data=validate_model_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            model_info = validate_model_response.data

            logger.info(f"Upload model successfully. Model ID: {model_info.ai_model_id}")

            return model_info

        except Exception as e:
            logger.error(f"Upload model failed. Error: {e}")
            raise e

    def get_model(self, model_id: str) -> ModelBase:
        self.token_handler.validate_token()

        try:
            logger.info("Getting model...")
            read_model_response = compressor_client_v2.read_model(
                ai_model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            model_info = read_model_response.data

            logger.info("Get model successfully.")

            return model_info

        except Exception as e:
            logger.error(f"Get model failed. Error: {e}")
            raise e

    def download_model(self, model_id: str, local_path: str) -> None:
        self.token_handler.validate_token()

        try:
            logger.info("Downloading model...")
            download_link = compressor_client_v2.download_model(
                ai_model_id=model_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            request.urlretrieve(download_link.data.presigned_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download model failed. Error: {e}")
            raise e

    def select_compression_method(
        self,
        model_id: str,
        compression_method: CompressionMethod,
        options: Optional[Options] = Options(),
    ) -> ResponseSelectMethod:
        """Select a compression method for a model.

        Args:
            model_id (str): The ID of the model.
            compression_method (CompressionMethod): The selected compression method.
            options(Options, optional): The options for pruning method.

        Raises:
            e: If an error occurs while selecting the compression method.

        Returns:
            ResponseSelectMethod: The compression information for the selected compression method.
        """

        self.token_handler.validate_token()

        try:
            logger.info("Selecting compression method...")

            get_available_layers_request = RequestAvailableLayers(
                compression_method=compression_method,
                options=options,
            )
            get_available_layers_response = compressor_client_v2.get_available_layers(
                ai_model_id=model_id,
                request_data=get_available_layers_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            available_layers_info = get_available_layers_response.data

            logger.info("Select compression method successfully.")

            return available_layers_info

        except Exception as e:
            logger.error(f"Select compression method failed. Error: {e}")
            raise e

    def get_compression(self, compression_id: str) -> ResponseCompression:
        self.token_handler.validate_token()

        try:
            logger.info("Getting compression...")
            read_compression_response = compressor_client_v2.read_compression(
                compression_id=compression_id,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            compression_info = read_compression_response.data

            logger.info("Get compression successfully.")

            return compression_info

        except Exception as e:
            logger.error(f"Get compression failed. Error: {e}")
            raise e

    def upload_dataset(self, compression_id: str, dataset_path: str) -> None:
        self.token_handler.validate_token()

        try:
            logger.info("Uploading dataset...")
            file_content = FileHandler.read_file_bytes(file_path=dataset_path)
            object_name = Path(dataset_path).name
            file = UploadFile(file_name=object_name, file_content=file_content)
            compressor_client_v2.upload_dataset(
                compression_id=compression_id,
                file=file,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            logger.info("Upload dataset successfully.")

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    def compress_model(
        self,
        compression: ResponseSelectMethod,
        output_dir: str,
        dataset_path: Optional[str] = None,
    ) -> CompressorMetadata:
        """Compress a model using the provided compression information.

        Args:
            compression (CompressionInfo): The information about the compression.
            output_dir (str): The local path to save the compressed model.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while compressing the model.

        Returns:
            CompressorMetadata: Compress metadata.
        """

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata: CompressorMetadata = self.initialize_metadata(
            output_dir=output_dir,
            input_model_path="",
            compression_method=compression.compression_method,
            ratio="",
            framework="",
            input_shapes=[],
        )

        try:
            logger.info("Compressing model...")
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            self.validate_token_and_check_credit(service_task=ServiceTask.ADVANCED_COMPRESSION)

            create_compression_request = RequestCreateCompression(
                ai_model_id=compression.input_model_id,
                compression_method=compression.compression_method,
                options=compression.options,
            )
            create_compression_response = compressor_client_v2.create_compression(
                request_data=create_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            for available_layers in compression.available_layers:
                if available_layers.values:
                    available_layers.use = True

            if dataset_path and compression.compression_method in [CompressionMethod.PR_NN, CompressionMethod.PR_SNP]:
                self.upload_dataset(create_compression_response.data.compression_id, dataset_path)

            update_compression_request = RequestUpdateCompression(
                available_layers=compression.available_layers,
                options=compression.options,
            )
            update_compression_response = compressor_client_v2.compress_model(
                compression_id=create_compression_response.data.compression_id,
                request_data=update_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            compression_info = update_compression_response.data
            model_info = self.get_model(model_id=compression.input_model_id)
            metadata.update_model_info(model_info.detail.framework, model_info.detail.input_layers)
            metadata = self.finalize_compression_process(metadata, model_info, compression_info, output_dir)

            self.print_remaining_credit(service_task=ServiceTask.ADVANCED_COMPRESSION)

            logger.info(f"Compress model successfully. Compressed Model ID: {compression_info.input_model_id}")

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.ADVANCED_COMPRESSION, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.ADVANCED_COMPRESSION)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

    def recommendation_compression(
        self,
        compression_method: CompressionMethod,
        recommendation_method: RecommendationMethod,
        recommendation_ratio: float,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        options: RecommendationOptions = RecommendationOptions(),
        dataset_path: Optional[str] = None,
    ) -> CompressorMetadata:
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
            CompressorMetadata: Compress metadata.
        """

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(
            output_dir=output_dir,
            input_model_path=input_model_path,
            compression_method=compression_method,
            ratio=recommendation_ratio,
            framework=framework,
            input_shapes=input_shapes,
        )

        try:
            logger.info("Compressing recommendation-based model...")
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            self.validate_token_and_check_credit(service_task=ServiceTask.ADVANCED_COMPRESSION)

            model_info = self.upload_model(input_model_path, input_shapes, framework)

            create_compression_request = RequestCreateCompression(
                ai_model_id=model_info.ai_model_id,
                compression_method=compression_method,
                options=options,
            )
            create_compression_response = compressor_client_v2.create_compression(
                request_data=create_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            if dataset_path and compression_method in [CompressionMethod.PR_NN, CompressionMethod.PR_SNP]:
                self.upload_dataset(create_compression_response.data.compression_id, dataset_path)

            logger.info("Calculating recommendation values...")
            create_recommendation_request = RequestCreateRecommendation(
                recommendation_method=recommendation_method,
                recommendation_ratio=recommendation_ratio,
                options=options,
            )
            create_recommendation_response = compressor_client_v2.create_recommendation(
                compression_id=create_compression_response.data.compression_id,
                request_data=create_recommendation_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            logger.info("Compressing model...")
            update_compression_request = RequestUpdateCompression(
                available_layers=create_recommendation_response.data.available_layers,
                options=options,
            )
            update_compression_response = compressor_client_v2.compress_model(
                compression_id=create_compression_response.data.compression_id,
                request_data=update_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            compression_info = update_compression_response.data
            metadata = self.finalize_compression_process(metadata, model_info, compression_info, output_dir)

            self.print_remaining_credit(service_task=ServiceTask.ADVANCED_COMPRESSION)

            logger.info(
                f"Recommendation compression successfully. Compressed Model ID: {compression_info.input_model_id}"
            )

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.ADVANCED_COMPRESSION, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.ADVANCED_COMPRESSION)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def get_input_model(self, input_model_id: str) -> Model:
        """Get model by ID.

        Args:
            input_model_id: ID of the model to retrieve
            user_id: User ID for verification

        Returns:
            Model object
        """
        with get_db_session() as db:
            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)
            return input_model

    def save_model(self, model_name: str, project_id: str, user_id: str) -> Model:
        """Create and save a new compressed model.

        Args:
            model_name: Name of the model
            project_id: Project ID to associate with the model
            user_id: User ID who owns the model

        Returns:
            Saved model object
        """
        model = Model(
            name=model_name,
            type=SubFolder.COMPRESSED_MODELS,
            is_retrainable=True,
            project_id=project_id,
            user_id=user_id,
        )
        return self._save_model(model)

    def _save_model(self, model: Model) -> Model:
        """Save model to database.

        Args:
            model: Model object to save

        Returns:
            Saved model with updated attributes
        """
        with get_db_session() as db:
            model = model_repository.save(db=db, model=model)
            return model

    def create_compression_task(
        self,
        compression_method: CompressionMethod,
        recommendation_ratio: float,
        options: Options,
        input_model_id: str,
        model_id: str,
        compression_task_id: str,
    ) -> CompressionTask:
        with get_db_session() as db:
            if compression_task_id:
                compression_task = CompressionTask(
                    task_id=compression_task_id,
                    method=compression_method,
                    ratio=recommendation_ratio,
                    options=options,
                    status=Status.NOT_STARTED,
                    input_model_id=input_model_id,
                    model_id=model_id,
                    user_id=self.user_info.user_id,
                )
            else:
                compression_task = CompressionTask(
                    method=compression_method,
                    ratio=recommendation_ratio,
                    options=options,
                    status=Status.NOT_STARTED,
                    input_model_id=input_model_id,
                    model_id=model_id,
                    user_id=self.user_info.user_id,
                )
            compression_task = compression_task_repository.save(db=db, model=compression_task)
            return compression_task

    def _save_compression_task(self, compression_task: CompressionTask) -> CompressionTask:
        """Save compression task to database.

        Args:
            compression_task: Compression task to save

        Returns:
            Saved compression task with updated attributes
        """
        with get_db_session() as db:
            compression_task = compression_task_repository.save(db=db, model=compression_task)
            return compression_task

    def get_training_task(self, model_id) -> TrainingTask:
        with get_db_session() as db:
            training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
            return training_task

    def _get_enum_value(self, enum_obj: Any) -> str:
        """Safely extract the string value from an enum or string.

        Args:
            enum_obj: Enum object or string

        Returns:
            String value of the enum or the original string
        """
        if hasattr(enum_obj, 'value'):
            return enum_obj.value
        return str(enum_obj)

    def recommendation_compression_from_id(
        self,
        input_model_id: str,
        compression_method: CompressionMethod,
        recommendation_method: RecommendationMethod,
        recommendation_ratio: float,
        framework: Framework = Framework.PYTORCH,
        options: RecommendationOptions = RecommendationOptions(),
        dataset_path: Optional[str] = None,
        compression_task_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        temp_dir = None

        try:
            self.validate_token_and_check_credit(service_task=ServiceTask.ADVANCED_COMPRESSION)

            # Handle output directory
            if output_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="netspresso_compression_")
                output_dir = temp_dir
            else:
                output_dir = FileHandler.create_unique_folder(folder_path=output_dir)

            input_model = self.get_input_model(input_model_id)
            project = self.get_project(project_id=input_model.project_id)

            # Download model to temporary directory
            download_dir = Path(output_dir) / "input_model"
            download_dir.mkdir(parents=True, exist_ok=True)

            remote_model_path = Path(input_model.object_path) / "model.pt"
            local_path = download_dir / "model.pt"

            logger.info(f"Downloading input model from Zenko: {remote_model_path}")
            storage_handler.download_file_from_s3(
                bucket_name=BUCKET_NAME,
                local_path=str(local_path),
                object_path=str(remote_model_path)
            )
            logger.info(f"Downloaded input model from Zenko: {local_path}")

            # Generate model name with safe enum value handling
            model_name_parts = [
                input_model.name,
                self._get_enum_value(compression_method),
                recommendation_ratio,
            ]
            model_name = "_".join(map(str, model_name_parts))
            logger.info(f"Model name: {model_name}")

            # Save compressed model
            model = self.save_model(
                model_name=model_name,
                project_id=input_model.project_id,
                user_id=self.user_info.user_id,
            )
            object_path = f"{project.user_id}/{project.project_id}/{model.model_id}/model.pt"
            logger.info(f"Object path: {object_path}")
            model.object_path = object_path
            model = self._save_model(model)

            # Create compression task
            compression_task = self.create_compression_task(
                compression_method=compression_method,
                recommendation_ratio=recommendation_ratio,
                options=options,
                input_model_id=input_model.model_id,
                model_id=model.model_id,
                compression_task_id=compression_task_id,
            )

            # Update status - Pass DB session
            compression_task.status = Status.IN_PROGRESS
            compression_task = self._save_compression_task(compression_task)

            training_task = self.get_training_task(input_model.model_id)
            model_info = self.upload_model(local_path, training_task.input_shapes, framework)

            create_compression_request = RequestCreateCompression(
                ai_model_id=model_info.ai_model_id,
                compression_method=compression_method,
                options=options,
            )
            create_compression_response = compressor_client_v2.create_compression(
                request_data=create_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            compression_task.compression_task_uuid = create_compression_response.data.compression_id
            compression_task = self._save_compression_task(compression_task)

            if dataset_path and compression_method in [CompressionMethod.PR_NN, CompressionMethod.PR_SNP]:
                remote_calibration_dataset_path = Path(input_model.object_path) / "calibration_dataset.npy"
                local_calibration_dataset_path = download_dir / "calibration_dataset.npy"

                logger.info(f"Downloading calibration dataset from Zenko: {remote_calibration_dataset_path}")
                storage_handler.download_file_from_s3(
                    bucket_name=BUCKET_NAME,
                    local_path=str(local_calibration_dataset_path),
                    object_path=str(remote_calibration_dataset_path)
                )
                logger.info(f"Downloaded calibration dataset from Zenko: {local_calibration_dataset_path}")
                self.upload_dataset(create_compression_response.data.compression_id, dataset_path)

            logger.info("Calculating recommendation values...")
            create_recommendation_request = RequestCreateRecommendation(
                recommendation_method=recommendation_method,
                recommendation_ratio=recommendation_ratio,
                options=options,
            )
            create_recommendation_response = compressor_client_v2.create_recommendation(
                compression_id=create_compression_response.data.compression_id,
                request_data=create_recommendation_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

            logger.info("Compressing model...")
            update_compression_request = RequestUpdateCompression(
                available_layers=create_recommendation_response.data.available_layers,
                options=options,
            )
            update_compression_response = compressor_client_v2.compress_model(
                compression_id=create_compression_response.data.compression_id,
                request_data=update_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            compression_info = update_compression_response.data

            compression_task.layers = [asdict(available_layer) for available_layer in compression_info.available_layers]
            compression_task = self._save_compression_task(compression_task)

            download_dir = Path(model.object_path).parent
            download_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading compressed model from Zenko: {download_dir}")

            # Download compressed model from Compressor Server
            self.download_model(compression_info.input_model_id, model.object_path)

            # Upload compressed model to Zenko
            storage_handler.upload_file_to_s3(
                bucket_name=BUCKET_NAME,
                local_path=model.object_path,
                object_path=model.object_path
            )

            if model_info.detail.framework in [Framework.PYTORCH, Framework.ONNX]:
                export_onnx(object_path, model_info.detail.input_layers)
                # Upload compressed onnx model to Zenko
                storage_handler.upload_file_to_s3(
                    bucket_name=BUCKET_NAME,
                    local_path=Path(model.object_path).with_suffix(".onnx").as_posix(),
                    object_path=Path(model.object_path).with_suffix(".onnx").as_posix()
                )

            logger.info(f"Uploaded Compressed Model file to Zenko: {model.object_path}")

            # Save model results for original and compressed models
            original_result = CompressionModelResult(
                size=model_info.file_size_in_mb,
                flops=model_info.detail.flops,
                number_of_parameters=model_info.detail.trainable_parameters + model_info.detail.non_trainable_parameters,
                trainable_parameters=model_info.detail.trainable_parameters,
                non_trainable_parameters=model_info.detail.non_trainable_parameters,
                number_of_layers=model_info.detail.number_of_layers if model_info.detail.number_of_layers != 0 else None,
                compression_task_id=compression_task.task_id,
                result_type='original'
            )

            compressed_model_info = self.get_model(model_id=compression_info.input_model_id)
            compressed_result = CompressionModelResult(
                size=compressed_model_info.file_size_in_mb,
                flops=compressed_model_info.detail.flops,
                number_of_parameters=compressed_model_info.detail.trainable_parameters + compressed_model_info.detail.non_trainable_parameters,
                trainable_parameters=compressed_model_info.detail.trainable_parameters,
                non_trainable_parameters=compressed_model_info.detail.non_trainable_parameters,
                number_of_layers=compressed_model_info.detail.number_of_layers if compressed_model_info.detail.number_of_layers != 0 else None,
                compression_task_id=compression_task.task_id,
                result_type='compressed'
            )

            self._save_compression_model_result(original_result)
            self._save_compression_model_result(compressed_result)

            self.print_remaining_credit(service_task=ServiceTask.ADVANCED_COMPRESSION)
            compression_task.status = Status.COMPLETED
            logger.info(
                f"Recommendation compression successfully. Compressed Model ID: {compression_info.input_model_id}"
            )

        except Exception as e:
            logger.error(f"Error in recommendation_compression_from_id: {e}")
            compression_task.status = Status.ERROR
            compression_task.error_detail = str(e) if e.args else "Unknown error"
            raise e
        finally:
            compression_task = self._save_compression_task(compression_task)

            # Clean up temporary directory (if output directory is a temporary directory)
            if temp_dir and os.path.exists(temp_dir) and output_dir == temp_dir:
                logger.info(f"Cleaning up temporary files in: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Successfully removed temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary files: {cleanup_error}")

        return compression_task.task_id

    def automatic_compression(
        self,
        input_model_path: str,
        output_dir: str,
        input_shapes: List[Dict[str, int]],
        framework: Framework = Framework.PYTORCH,
        compression_ratio: float = 0.5,
    ) -> CompressorMetadata:
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
            CompressorMetadata: Compress metadata.
        """
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        metadata = self.initialize_metadata(
            output_dir=output_dir,
            input_model_path=input_model_path,
            compression_method=CompressionMethod.PR_L2,
            ratio=compression_ratio,
            framework=framework,
            input_shapes=input_shapes,
        )

        try:
            logger.info("Compressing automatic-based model...")
            if metadata.status in [Status.ERROR, Status.STOPPED]:
                return metadata

            model_info = self.upload_model(input_model_path, input_shapes, framework)

            logger.info("Compressing model...")
            automatic_compression_request = RequestAutomaticCompressionParams(compression_ratio=compression_ratio)
            automatic_compression_response = compressor_client_v2.compress_model_with_automatic(
                ai_model_id=model_info.ai_model_id,
                request_data=automatic_compression_request,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            compression_info = automatic_compression_response.data
            metadata = self.finalize_compression_process(metadata, model_info, compression_info, output_dir)

            self.print_remaining_credit(service_task=ServiceTask.AUTOMATIC_COMPRESSION)

            logger.info(f"Automatic compression successfully. Compressed Model ID: {compression_info.input_model_id}")

        except Exception as e:
            metadata = self.handle_error(metadata, ServiceTask.AUTOMATIC_COMPRESSION, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.AUTOMATIC_COMPRESSION)
        finally:
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def _save_compression_model_result(self, result: CompressionModelResult) -> CompressionModelResult:
        with get_db_session() as db:
            result = compression_model_result_repository.save(db=db, model=result)
            return result
