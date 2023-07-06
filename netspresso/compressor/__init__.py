from functools import wraps
from typing import Dict, List, Union

from urllib import request
from loguru import logger

from netspresso.compressor.client import (
    ModelCompressorAPIClient,
    Task,
    Framework,
    CompressionMethod,
    RecommendationMethod,
)
from netspresso.compressor.client.schemas.auth import LoginRequest, RefreshTokenRequest
from netspresso.compressor.client.schemas.model import UploadModelRequest
from netspresso.compressor.client.schemas.compression import (
    AutoCompressionRequest,
    CompressionRequest,
    GetAvailableLayersRequest,
    CreateCompressionRequest,
    RecommendationRequest,
    UploadDatasetRequest,
)
from netspresso.compressor.core.model import CompressedModel, Model, ModelCollection
from netspresso.compressor.core.compression import CompressionInfo
from netspresso.compressor.utils.model import (
    get_compressed_model_object,
    get_model_collection_object,
    get_model_object,
)
from netspresso.compressor.utils.token import check_jwt_exp


class ModelCompressor:
    def __init__(self, email: str, password: str):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """

        self.email = email
        self.password = password
        self.client = ModelCompressorAPIClient()
        self.__login()

    def __login(self) -> None:
        try:
            data = LoginRequest(username=self.email, password=self.password)
            response = self.client.login(data)
            self.access_token = response.access_token
            self.refresh_token = response.refresh_token
            logger.info("Login successful")

        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    def validate_token(func) -> None:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not check_jwt_exp(self.access_token):
                self.__reissue_token()
            return func(self, *args, **kwargs)

        return wrapper

    def __reissue_token(self) -> None:
        try:
            data = RefreshTokenRequest(access_token=self.access_token, refresh_token=self.refresh_token)
            response = self.client.refresh_token(data)
            self.access_token = response.access_token
            self.refresh_token = response.refresh_token

        except Exception as e:
            raise e

    @validate_token
    def get_credit(self) -> int:
        """Get the available NetsPresso credits.

        Raises:
            e: If an error occurs while getting credit information.

        Returns:
            int: The total amount of available NetsPresso credits.
        """

        try:
            credit = self.client.get_credit(access_token=self.access_token)
            logger.info(f"Get Credit successful. Credit: {credit.total}")

            return credit.total

        except Exception as e:
            logger.error(f"Get Credit failed. Error: {e}")
            raise e

    @validate_token
    def upload_model(
        self, model_name: str, task: Task, framework: Framework, file_path: str, input_shapes: List[Dict[str, int]] = []
    ) -> Model:
        """Upload a model for compression.

        Args:
            model_name (str): The name of the model.
            task (Task): The task of the model.
            framework (Framework): The framework of the model.
            file_path (str): The file path where the model is located.
            input_shapes (List[Dict[str, int]], optional): Input shapes of the model. Defaults to [].

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            Model: Uploaded model object.
        """

        try:
            logger.info("Uploading Model...")
            data = UploadModelRequest(
                model_name=model_name,
                task=task,
                framework=framework,
                file_path=file_path,
                input_layers=input_shapes,
            )
            model_info = self.client.upload_model(data=data, access_token=self.access_token)
            model = get_model_object(model_info=model_info)

            logger.info(f"Upload model successful. Model ID: {model.model_id}")

            return model

        except Exception as e:
            logger.error(f"Upload model failed. Error: {e}")
            raise e

    @validate_token
    def get_models(self) -> List[ModelCollection]:
        """Get the list of uploaded & compressed models.

        Raises:
            e: If an error occurs while getting the model list.

        Returns:
            List[ModelCollection]: The list of uploaded & compressed models.
        """

        try:
            logger.info("Getting model list...")
            models = []
            parent_models = self.client.get_parent_models(is_simple=True, access_token=self.access_token)

            for parent_model_info in parent_models:
                if parent_model_info.origin_from == "custom":
                    model = get_model_collection_object(model_info=parent_model_info)

                    children_models = self.client.get_children_models(
                        model_id=model.model_id, access_token=self.access_token
                    )
                    model.compressed_models = [
                        get_compressed_model_object(children_model) for children_model in children_models
                    ]

                models.append(model)
            logger.info("Get model list successful.")

            return models

        except Exception as e:
            logger.error(f"Get model list failed. Error: {e}")
            raise e

    @validate_token
    def get_uploaded_models(self) -> List[Model]:
        """Get the list of uploaded models.

        Raises:
            e: If an error occurs while getting the uploaded model list.

        Returns:
            List[Model]: The list of uploaded models.
        """

        try:
            logger.info("Getting uploaded model list...")
            parent_models = self.client.get_parent_models(is_simple=True, access_token=self.access_token)
            uploaded_models = [
                get_model_object(model_info=parent_model_info)
                for parent_model_info in parent_models
                if parent_model_info.origin_from == "custom"
            ]
            logger.info("Get uploaded model list successful.")

            return uploaded_models

        except Exception as e:
            logger.error(f"Get uploaded model list failed. Error: {e}")
            raise e

    @validate_token
    def get_compressed_models(self, model_id: str) -> List[CompressedModel]:
        """Get the list of compressed models for a given model ID.

        Args:
            model_id (str): The ID of the model.

        Raises:
            e: If an error occurs while getting the compressed model list.

        Returns:
            List[CompressedModel]: The list of compressed models for a given model ID.
        """

        try:
            logger.info("Getting compressed model list...")
            children_models = self.client.get_children_models(model_id=model_id, access_token=self.access_token)
            compressed_models = [
                get_compressed_model_object(model_info=children_model_info) for children_model_info in children_models
            ]
            logger.info("Get compressed model list successful.")

            return compressed_models

        except Exception as e:
            logger.error(f"Get compressed model list failed. Error: {e}")
            raise e

    @validate_token
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

        try:
            logger.info("Getting model...")
            model_info = self.client.get_model_info(model_id=model_id, access_token=self.access_token)
            if model_info.status.is_compressed:
                model = get_compressed_model_object(model_info=model_info)
            else:
                model = get_model_object(model_info=model_info)
            logger.info("Get model successful.")

            return model

        except Exception as e:
            logger.error(f"Get model failed. Error: {e}")
            raise e

    @validate_token
    def download_model(self, model_id: str, local_path: str) -> None:
        """Download the model for a given model ID to the local path.

        Args:
            model_id (str): The ID of the model.
            local_path (str): The local path to save the downloaded model.

        Raises:
            e: If an error occurs while downloading the model.
        """

        try:
            logger.info("Downloading model...")
            download_link = self.client.get_download_model_link(model_id=model_id, access_token=self.access_token)
            request.urlretrieve(download_link.url, local_path)
            logger.info(f"Download model successful. Local Path: {local_path}")

        except Exception as e:
            logger.error(f"Download model failed. Error: {e}")
            raise e

    @validate_token
    def delete_model(self, model_id: str, recursive: bool = False) -> None:
        """Delete the model for a given model ID.

        Args:
            model_id (str): The ID of the model.
            recursive (bool, optional): Whether to also delete the compressed model for that model. Defaults to False.

        Raises:
            e: If an error occurs while deleting the model.
        """

        try:
            logger.info("Deleting model...")
            children_models = self.client.get_children_models(model_id=model_id, access_token=self.access_token)
            if len(children_models) != 0:
                if not recursive:
                    logger.warning(
                        "Deleting the model will also delete its compressed models. To proceed with the deletion, set the `recursive` parameter to True."
                    )
                else:
                    logger.info("The compressed model for that model will also be deleted.")
                    self.client.delete_model(model_id=model_id, access_token=self.access_token)
                    logger.info("Delete model successful.")
            else:
                logger.info("The model will be deleted.")
                self.client.delete_model(model_id=model_id, access_token=self.access_token)
                logger.info("Delete model successful.")

        except Exception as e:
            logger.error(f"Delete model failed. Error: {e}")
            raise e

    @validate_token
    def select_compression_method(self, model_id: str, compression_method: CompressionMethod) -> CompressionInfo:
        """Select a compression method for a model.

        Args:
            model_id (str): The ID of the model.
            compression_method (CompressionMethod): The selected compression method.

        Raises:
            e: If an error occurs while selecting the compression method.

        Returns:
            CompressionInfo: The compression information for the selected compression method.
        """
        try:
            logger.info("Selecting compression method...")
            data = GetAvailableLayersRequest(model_id=model_id, compression_method=compression_method)
            response = self.client.get_available_layers(data=data, access_token=self.access_token)
            compression_info = CompressionInfo(
                original_model_id=model_id,
                compression_method=compression_method,
                available_layers=response.available_layers,
            )
            logger.info("Select compression method successful.")

            return compression_info

        except Exception as e:
            logger.error(f"Select compression method failed. Error: {e}")
            raise e

    @validate_token
    def get_compression(self, compression_id: str) -> CompressionInfo:
        """Get information about a compression.

        Args:
            compression_id (str): The ID of the compression.

        Raises:
            e: If an error occurs while getting the compression information.

        Returns:
            CompressionInfo: The information about the compression.
        """
        try:
            logger.info("Getting compression...")
            compression_info = self.client.get_compression_info(
                compression_id=compression_id, access_token=self.access_token
            )
            compression_info = CompressionInfo(
                compressed_model_id=compression_info.new_model_id,
                compression_id=compression_info.compression_id,
                compression_method=compression_info.compression_method,
                available_layers=compression_info.available_layers,
            )
            logger.info("Get compression successful.")

            return compression_info

        except Exception as e:
            logger.error(f"Get compression failed. Error: {e}")
            raise e

    @validate_token
    def __upload_dataset(self, model_id: str, dataset_path: str) -> None:
        """Upload a dataset for nuclear norm compression method.

        Args:
            model_id (str): The ID of the model.
            dataset_path (str): The file path where the dataset is located.

        Raises:
            e: If an error occurs while uploading the dataset.
        """
        try:
            logger.info(f"Uploading dataset...")
            data = UploadDatasetRequest(model_id=model_id, file_path=dataset_path)
            self.client.upload_dataset(data=data, access_token=self.access_token)
            logger.info(f"Upload dataset successful.")

        except Exception as e:
            logger.error(f"Upload dataset failed. Error: {e}")
            raise e

    @validate_token
    def compress_model(
        self, compression: CompressionInfo, model_name: str, output_path: str, dataset_path: str = None
    ) -> CompressedModel:
        """Compress a model using the provided compression information.

        Args:
            compression (CompressionInfo): The information about the compression.
            model_name (str): The name of the compressed model.
            output_path (str): The local path to save the compressed model.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while compressing the model.

        Returns:
            CompressedModel: The compressed model.
        """
        try:
            logger.info("Compressing model...")
            data = CreateCompressionRequest(
                model_id=compression.original_model_id,
                model_name=model_name,
                compression_method=compression.compression_method,
            )
            compression_info = self.client.create_compression(data=data, access_token=self.access_token)

            if dataset_path and compression.compression_method == CompressionMethod.PR_NN:
                self.__upload_dataset(model_id=compression.original_model_id, dataset_path=dataset_path)

            for available_layers in compression.available_layers:
                if available_layers.values != [""]:
                    available_layers.use = True

            all_layers_false = all(available_layer.values == [""] for available_layer in compression.available_layers)
            if all_layers_false:
                raise Exception(
                    f"The available_layer.values all empty. please put in the available_layer.values to compress."
                )

            data = CompressionRequest(
                compression_id=compression_info.compression_id,
                compression_method=compression.compression_method,
                layers=compression.available_layers,
                compressed_model_id=compression_info.new_model_id,
            )
            self.client.compress_model(data=data, access_token=self.access_token)
            self.download_model(model_id=compression_info.new_model_id, local_path=output_path)
            compressed_model = self.get_model(model_id=compression_info.new_model_id)
            logger.info(f"Compress model successful. Compressed Model ID: {compressed_model.model_id}")
            logger.info("50 credits have been consumed.")

            return compressed_model

        except Exception as e:
            logger.error(f"Compress model failed. Error: {e}")
            raise e

    @validate_token
    def recommendation_compression(
        self,
        model_id: str,
        model_name: str,
        compression_method: CompressionMethod,
        recommendation_method: RecommendationMethod,
        recommendation_ratio: float,
        output_path: str,
        dataset_path: str = None,
    ) -> CompressedModel:
        """Compress a recommendation-based model using the given compression and recommendation methods.

        Args:
            model_id (str): The ID of the model.
            model_name (str): The name of the compressed model.
            compression_method (CompressionMethod): The selected compression method.
            recommendation_method (RecommendationMethod): The selected recommendation method.
            recommendation_ratio (float): The compression ratio recommended by the recommendation method.
            output_path (str): The local path to save the compressed model.
            dataset_path (str, optional): The path of the dataset used for nuclear norm compression method. Default is None.

        Raises:
            e: If an error occurs while performing recommendation compression.

        Returns:
            CompressedModel: The compressed model.
        """

        try:
            logger.info("Compressing recommendation-based model...")

            if compression_method in [CompressionMethod.PR_ID, CompressionMethod.FD_CP]:
                raise Exception(
                    f"The {compression_method} compression method you choose doesn't provide a recommendation."
                )

            if (
                compression_method in [CompressionMethod.PR_L2, CompressionMethod.PR_GM, CompressionMethod.PR_NN]
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

            data = CreateCompressionRequest(
                model_id=model_id,
                model_name=model_name,
                compression_method=compression_method,
            )
            compression_info = self.client.create_compression(data=data, access_token=self.access_token)

            if dataset_path and compression_method == CompressionMethod.PR_NN:
                self.__upload_dataset(model_id=model_id, dataset_path=dataset_path)

            data = RecommendationRequest(
                model_id=model_id,
                compression_id=compression_info.compression_id,
                recommendation_method=recommendation_method,
                recommendation_ratio=recommendation_ratio,
            )
            recommendation_result = self.client.get_recommendation(data=data, access_token=self.access_token)

            for available_layer, recommended_info in zip(
                compression_info.available_layers, recommendation_result.recommended_layers
            ):
                available_layer.use = True
                available_layer.values = recommended_info.values

            data = CompressionRequest(
                compression_id=compression_info.compression_id,
                compression_method=compression_method,
                layers=compression_info.available_layers,
                compressed_model_id=compression_info.new_model_id,
            )
            self.client.compress_model(data=data, access_token=self.access_token)
            self.download_model(model_id=compression_info.new_model_id, local_path=output_path)
            compressed_model = self.get_model(model_id=compression_info.new_model_id)
            logger.info(f"Recommendation compression successful. Compressed Model ID: {compressed_model.model_id}")
            logger.info("50 credits have been consumed.")

            return compressed_model

        except Exception as e:
            logger.error(f"Recommendation compression failed. Error: {e}")
            raise e

    @validate_token
    def automatic_compression(
        self, model_id: str, model_name: str, output_path: str, compression_ratio: float = 0.5
    ) -> CompressedModel:
        """Compress a model automatically based on the given compression ratio.

        Args:
            model_id (str): The ID of the model.
            model_name (str): The name of the compressed model.
            output_path (str): The local path to save the compressed model.
            compression_ratio (float): The compression ratio for automatic compression. Defaults to 0.5.

        Raises:
            e: If an error occurs while performing automatic compression.

        Returns:
            CompressedModel: The compressed model.
        """

        try:
            logger.info("Compressing automatic-based model...")
            data = AutoCompressionRequest(
                model_id=model_id,
                model_name=model_name,
                recommendation_ratio=compression_ratio,
                save_path=output_path,
            )
            model_info = self.client.auto_compression(data=data, access_token=self.access_token)
            self.download_model(model_id=model_info.model_id, local_path=output_path)
            compressed_model = get_compressed_model_object(model_info=model_info)
            logger.info(f"Automatic compression successful. Compressed Model ID: {compressed_model.model_id}")
            logger.info("25 credits have been consumed.")

            return compressed_model

        except Exception as e:
            logger.error(f"Automatic compression failed. Error: {e}")
            raise e
