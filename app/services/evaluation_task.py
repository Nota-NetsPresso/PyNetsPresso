import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.orm import Session

from app.api.v1.schemas.device import (
    HardwareTypePayload,
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
)
from app.api.v1.schemas.task.conversion.conversion_task import (
    TargetFrameworkPayload,
)
from app.api.v1.schemas.task.evaluation.evaluation_task import (
    BoundingBox,
    EvaluationCreate,
    EvaluationPayload,
    EvaluationResultsPayload,
    ImagePrediction,
    PredictionForThreshold,
)
from app.exceptions.evaluation import EvaluationTaskAlreadyExistsException
from app.services.project import project_service
from app.worker.evaluation_task import chain_conversion_and_evaluation, run_multiple_evaluations
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums import DataType, DeviceName, SoftwareVersion, Status
from netspresso.enums.conversion import EvaluationTargetFramework, SourceFramework, TargetFramework
from netspresso.evaluator.evaluator import EVALUATION_BUCKET_NAME
from netspresso.exceptions.conversion import ConversionTaskNotFoundException
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.models.evaluation import EvaluationDataset, EvaluationTask
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.evaluation import evaluation_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.file import FileHandler

storage_handler = ObjectStorageHandler()


class EvaluationTaskService:
    def get_supported_devices(
        self, db: Session, framework: SourceFramework, api_key: str
    ) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            framework (SourceFramework): Framework to get supported devices for
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        netspresso = NetsPresso(api_key=api_key)
        converter = netspresso.converter_v2()
        supported_options = converter.get_supported_options(framework=framework)

        supported_framework = [TargetFramework.TENSORFLOW_LITE]

        return [self._create_supported_device_response(option) for option in supported_options if option.framework in supported_framework]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        return SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[self._create_device_payload(device) for device in option.devices],
        )

    def _create_device_payload(self, device: DeviceInfo) -> SupportedDevicePayload:
        """Create SupportedDevicePayload from device information.

        Args:
            device: Device information containing name, versions, precisions, and hardware types

        Returns:
            SupportedDevicePayload: Payload containing device information
        """
        return SupportedDevicePayload(
            name=device.device_name,
            software_versions=[
                SoftwareVersionPayload(name=version.software_version) for version in device.software_versions
            ],
            precisions=[PrecisionForConversionPayload(name=precision) for precision in device.data_types],
            hardware_types=[HardwareTypePayload(name=hardware_type) for hardware_type in device.hardware_types],
        )

    def _find_existing_conversion_task(
        self,
        db: Session,
        input_model_id: str,
        target_framework: TargetFramework,
        target_device_name: DeviceName,
        target_software_version: Optional[SoftwareVersion] = None,
        target_data_type: DataType = DataType.FP16
    ) -> Optional[ConversionTask]:
        """Find an existing conversion task that matches the given parameters.

        Args:
            input_model_id: ID of the input model
            target_framework: Target framework for conversion
            target_device_name: Target device for conversion
            target_software_version: Target software version (optional)
            target_data_type: Target data type/precision

        Returns:
            The task_id of the matching conversion task, or None if no match found
        """
        # Find conversion tasks for the input model
        conversion_tasks = conversion_task_repository.get_all_by_model_id(
            db=db,
            model_id=input_model_id
        )

        # Filter tasks by the conversion parameters
        for task in conversion_tasks:
            if (task.framework == target_framework and
                task.device_name == target_device_name and
                task.precision == target_data_type and
                (target_software_version is None or task.software_version == target_software_version) and
                task.status == Status.COMPLETED):
                return task

        raise ConversionTaskNotFoundException()

    def _check_evaluation_task_status(self, db: Session, model_id: str, dataset_path: str, confidence_score: float):
        evaluation_task = evaluation_task_repository.get_by_model_dataset_and_confidence(
            db=db,
            model_id=model_id,
            dataset_id=dataset_path,
            confidence_score=confidence_score
        )

        if evaluation_task:
            if evaluation_task.status == Status.COMPLETED:
                logger.warning(f"Evaluation task already completed: {evaluation_task.task_id}")
                raise EvaluationTaskAlreadyExistsException(task_id=evaluation_task.task_id, task_status=Status.COMPLETED.value)
            elif evaluation_task.status == Status.IN_PROGRESS:
                logger.warning(f"Evaluation task already in progress: {evaluation_task.task_id}")
                raise EvaluationTaskAlreadyExistsException(task_id=evaluation_task.task_id, task_status=Status.IN_PROGRESS.value)
            elif evaluation_task.status == Status.ERROR:
                logger.info(f"Retrying failed evaluation task: {evaluation_task.task_id}")
            else:
                # Other status (NOT_STARTED, STOPPED, etc.)
                logger.info(f"Using existing evaluation task with ID: {evaluation_task.task_id}")

    def create_evaluation_task(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        confidence_scores = [0.3, 0.5, 0.6]

        if evaluation_in.conversion.framework == EvaluationTargetFramework.ONNX:
            logger.info("Processing ONNX model evaluation without conversion")

            # Get model information
            model = model_repository.get_by_model_id(
                db=db,
                model_id=evaluation_in.input_model_id
            )

            try:
                for confidence_score in confidence_scores:
                    self._check_evaluation_task_status(
                        db=db,
                        model_id=model.model_id,
                        dataset_path=evaluation_in.dataset_path,
                        confidence_score=confidence_score
                    )
            except EvaluationTaskAlreadyExistsException:
                raise

            task_result = run_multiple_evaluations.apply_async(
                kwargs={
                    "api_key": api_key,
                    "model_id": model.model_id,
                    "dataset_id": evaluation_in.dataset_path,
                    "training_task_id": evaluation_in.training_task_id,
                    "confidence_scores": confidence_scores,
                },
            )

            evaluation_task_id = task_result.get(timeout=5)
            logger.info(f"ONNX evaluation task ID: {evaluation_task_id}")

            return evaluation_task_id

        try:
            # Check if a conversion task exists
            conversion_task = self._find_existing_conversion_task(
                db=db,
                input_model_id=evaluation_in.input_model_id,
                target_framework=evaluation_in.conversion.framework,
                target_device_name=evaluation_in.conversion.device_name,
                target_software_version=evaluation_in.conversion.software_version,
                target_data_type=evaluation_in.conversion.precision
            )

            # If conversion task exists, start only the evaluation
            try:
                for confidence_score in confidence_scores:
                    self._check_evaluation_task_status(
                        db=db,
                        model_id=conversion_task.model_id,
                        dataset_path=evaluation_in.dataset_path,
                        confidence_score=confidence_score
                    )
            except EvaluationTaskAlreadyExistsException:
                raise

            task_result = run_multiple_evaluations.apply_async(
                kwargs={
                    "api_key": api_key,
                    "model_id": conversion_task.model_id,
                    "dataset_id": evaluation_in.dataset_path,
                    "training_task_id": evaluation_in.training_task_id,
                    "confidence_scores": confidence_scores,
                },
            )

            evaluation_task_id = task_result.get(timeout=5)
            logger.info(f"Evaluation task ID: {evaluation_task_id}")

            return evaluation_task_id

        except ConversionTaskNotFoundException:
            # If no conversion task exists, chain conversion and evaluation together
            logger.info("No existing conversion task found. Creating a new conversion task and chaining with evaluation.")

            # Get model information
            model = model_repository.get_by_model_id(
                db=db,
                model_id=evaluation_in.input_model_id
            )

            if not model:
                raise Exception(f"Input model with ID {evaluation_in.input_model_id} not found")

            # Get project information
            project = project_service.get_project(db=db, project_id=model.project_id, api_key=api_key)

            # Create input model and output directory paths
            project_abs_path = Path(project.project_abs_path)
            input_model_dir = project_abs_path / model.object_path

            input_model_path = input_model_dir / "model.onnx"
            output_dir = input_model_dir / "converted"

            logger.info(f"Input model path: {input_model_path}")
            logger.info(f"Output directory: {output_dir}")

            task_result = chain_conversion_and_evaluation.apply_async(
                kwargs={
                    "api_key": api_key,
                    "input_model_path": input_model_path.as_posix(),
                    "output_dir": output_dir.as_posix(),
                    "target_framework": evaluation_in.conversion.framework,
                    "target_device_name": evaluation_in.conversion.device_name,
                    "target_data_type": evaluation_in.conversion.precision,
                    "target_software_version": evaluation_in.conversion.software_version,
                    "input_layer": None,
                    "dataset_path": None,
                    "input_model_id": evaluation_in.input_model_id,
                    "dataset_id": evaluation_in.dataset_path,
                    "training_task_id": evaluation_in.training_task_id,
                    "confidence_scores": confidence_scores,
                }
            )

            # Get the starting task ID of the chain
            evaluation_task_id = task_result.get(timeout=5)
            logger.info(f"Conversion and evaluation chain started with ID: {evaluation_task_id}")

            return evaluation_task_id

    def get_evaluation_tasks(
        self,
        db: Session,
        api_key: str,
        model_id: str,
    ) -> List[EvaluationPayload]:
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()
        evaluation_tasks = evaluator.get_evaluation_tasks(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=model_id
        )

        return [EvaluationPayload.model_validate(evaluation_task) for evaluation_task in evaluation_tasks]

    def count_evaluation_task_by_user_id(
        self,
        db: Session,
        api_key: str,
        model_id: str,
    ) -> int:
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()

        return evaluator.count_evaluation_task_by_user_id(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=model_id
        )

    def get_unique_datasets_by_model_id(
        self,
        db: Session,
        api_key: str,
        model_id: str,
    ) -> List[EvaluationDataset]:
        """Get unique dataset IDs used for evaluating a specific model.

        Args:
            db: Database session
            api_key: API key for authentication
            model_id: Model ID

        Returns:
            List[str]: List of unique dataset IDs
        """
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()

        return evaluator.get_unique_datasets_by_model_id(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=model_id
        )

    def get_evaluation_results_by_model_and_dataset(
        self,
        db: Session,
        api_key: str,
        model_id: str,
        dataset_id: str,
    ) -> List[EvaluationTask]:
        """Get evaluation results for a specific model and dataset.

        Args:
            db: Database session
            api_key: API key for authentication
            model_id: Model ID
            dataset_id: Dataset ID

        Returns:
            List[EvaluationTask]: List of evaluation results
        """
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()

        evaluation_tasks = evaluator.get_completed_evaluation_results_by_model_and_dataset(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=model_id,
            dataset_id=dataset_id
        )

        return evaluation_tasks

    def get_image_urls_from_s3(self, user_id: str, task_id: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Get image paths and presigned URLs from S3 for a specific task

        Args:
            user_id: User ID
            task_id: Task ID

        Returns:
            Tuple containing list of image paths and dictionary of image URLs
        """
        # Get list of image files from result_images directory
        result_images_prefix = f"{user_id}/{task_id}/result_images/"
        image_objects = storage_handler.list_objects(
            bucket_name=EVALUATION_BUCKET_NAME,
            prefix=result_images_prefix
        )

        # Filter only files with '_images' in the path
        image_paths = [obj for obj in image_objects if '_images' in Path(obj).name]

        # Create presigned URLs for each image
        image_urls = {}
        for image_path in image_paths:
            # Extract image filename (e.g., "000001_images.png")
            image_filename = Path(image_path).name

            # Create presigned URL
            presigned_url = storage_handler.get_download_presigned_url(
                bucket_name=EVALUATION_BUCKET_NAME,
                object_path=image_path,
                download_name=image_filename,
                expires_in=3600  # 1 hour
            )

            # Store URL with filename as key
            image_urls[image_filename] = presigned_url

        return image_paths, image_urls

    def _initialize_image_predictions(self, image_paths: List[str], image_urls: Dict[str, str]) -> Dict[str, ImagePrediction]:
        """
        Initialize ImagePrediction objects from a list of image paths

        Args:
            image_paths: List of image paths
            image_urls: Dictionary of image URLs (filename -> URL)

        Returns:
            Dictionary of ImagePrediction objects for each image
        """
        image_predictions = {}

        for image_path in image_paths:
            image_filename = Path(image_path).name
            image_url = image_urls.get(image_filename)

            if not image_url:
                logger.warning(f"No presigned URL found for image {image_filename}")
                continue

            # Create prediction entry for this image
            image_predictions[image_filename] = ImagePrediction(
                image_id=image_filename,
                image_url=image_url,
                predictions=[]
            )

        return image_predictions

    def _process_threshold_predictions(
        self,
        threshold: float,
        task: EvaluationTask,
        image_paths: List[str],
        image_predictions: Dict[str, ImagePrediction],
        temp_path: Path
    ) -> None:
        """
        Process predictions for a specific threshold value

        Args:
            threshold: Threshold value to process
            task: Evaluation task
            image_paths: List of image paths
            image_predictions: Dictionary of image predictions to update
            temp_path: Temporary file path
        """
        user_id = task.user_id
        task_id = task.task_id

        # Download predictions.json for this threshold
        predictions_object_path = f"{user_id}/{task_id}/predictions.json"
        predictions_local_path = temp_path / f"predictions_{threshold}.json"

        try:
            storage_handler.download_file_from_s3(
                bucket_name=EVALUATION_BUCKET_NAME,
                object_path=predictions_object_path,
                local_path=predictions_local_path.as_posix()
            )

            # Read predictions.json
            predictions_data = FileHandler.load_json(predictions_local_path)

            # Process predictions for this threshold
            if "predictions" not in predictions_data:
                logger.warning(f"No predictions found in {predictions_local_path}")
                return

            base_predictions = predictions_data["predictions"]

            # Limit to the minimum length of both arrays to ensure 1:1 mapping
            max_items = min(len(base_predictions), len(image_paths))

            # Process each image prediction
            for i in range(max_items):
                pred = base_predictions[i]
                image_path = image_paths[i]

                # Extract filename from path
                image_filename = Path(image_path).name

                # Skip if this image was not initialized (likely due to missing URL)
                if image_filename not in image_predictions:
                    continue

                # Add predictions for this threshold to the image
                bboxes = [BoundingBox.model_validate(bbox) for bbox in pred.get("bboxes", [])]

                # Create threshold-specific prediction
                threshold_prediction = PredictionForThreshold(
                    threshold=threshold,
                    bboxes=bboxes
                )

                # Add to the image's predictions list
                image_predictions[image_filename].predictions.append(threshold_prediction)

        except Exception as e:
            logger.error(f"Error processing predictions for threshold {threshold}: {str(e)}")
            # Continue with other thresholds even if one fails

    def get_evaluation_result_details(
        self,
        db: Session,
        api_key: str,
        converted_model_id: str,
        dataset_id: str,
        start: int = 0,
        size: int = 20,
    ) -> EvaluationResultsPayload:
        """Get detailed evaluation results including predictions and result images with pagination.

        Args:
            db: Database session
            api_key: API key for authentication
            model_id: Model ID
            dataset_id: Dataset ID
            start: Pagination start index
            size: Page size (number of images)

        Returns:
            EvaluationResultsPayload: Detailed evaluation results with predictions and image URLs
        """
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()

        # Get evaluation tasks for this model and dataset
        evaluation_tasks = evaluator.get_completed_evaluation_results_by_model_and_dataset(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=converted_model_id,
            dataset_id=dataset_id
        )

        if not evaluation_tasks:
            logger.warning(f"No evaluation tasks found for model {converted_model_id} and dataset {dataset_id}")
            return EvaluationResultsPayload(
                model_id=converted_model_id,
                dataset_id=dataset_id,
                results=[],
                result_count=0,
                total_count=0
            )

        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="evaluation_results_")
        temp_path = Path(temp_dir)

        logger.info(f"evaluation_tasks: {len(evaluation_tasks)}")

        try:
            # Get list of image files and presigned URLs using the first completed task
            # (image URLs should be the same for all tasks with the same dataset)
            first_task = evaluation_tasks[0]
            image_paths, image_urls = self.get_image_urls_from_s3(first_task.user_id, first_task.task_id)

            # Sort image paths to ensure consistent ordering
            image_paths.sort()

            # 1. Initialize prediction objects for all images
            image_predictions = self._initialize_image_predictions(image_paths, image_urls)

            # 2. Process each evaluation task directly
            for task in evaluation_tasks:
                # Use the actual threshold from the database
                threshold = task.confidence_score

                logger.info(f"Processing task with confidence score {threshold}")
                self._process_threshold_predictions(
                    threshold=threshold,
                    task=task,
                    image_paths=image_paths,
                    image_predictions=image_predictions,
                    temp_path=temp_path
                )

            # 3. Convert results and apply pagination
            results = list(image_predictions.values())

            total_count = len(results)

            # Check for empty results
            if total_count == 0:
                logger.warning("No image predictions were created")
                return EvaluationResultsPayload(
                    model_id=converted_model_id,
                    dataset_id=dataset_id,
                    results=[],
                    result_count=0,
                    total_count=0
                )

            # Validate start index
            if start >= total_count:
                start = 0

            # Calculate end index
            end = min(start + size, total_count)

            # Get paginated results
            paginated_predictions = results[start:end]

            # Return combined results with pagination info
            return EvaluationResultsPayload(
                model_id=converted_model_id,
                dataset_id=dataset_id,
                results=paginated_predictions,
                result_count=len(paginated_predictions),
                total_count=total_count
            )

        except Exception as e:
            logger.error(f"Error retrieving evaluation results: {str(e)}")
            raise e

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def delete_evaluation_task(self, db: Session, evaluation_task_id: str, api_key: str) -> EvaluationPayload:
        evaluation_task = evaluation_task_repository.get_by_task_id(db=db, task_id=evaluation_task_id)
        evaluation_task_repository.soft_delete(db=db, model=evaluation_task)

        return EvaluationPayload.model_validate(evaluation_task)


evaluation_task_service = EvaluationTaskService()
