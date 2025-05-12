import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

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
    EvaluationCreate,
    EvaluationPayload,
    EvaluationResultsPayload,
)
from app.exceptions.evaluation import EvaluationTaskAlreadyExistsException
from app.worker.evaluation_task import run_multiple_evaluations
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums import DataType, DeviceName, SoftwareVersion, Status
from netspresso.enums.conversion import SourceFramework, TargetFramework
from netspresso.evaluator.evaluator import EVALUATION_BUCKET_NAME
from netspresso.exceptions.conversion import ConversionTaskNotFoundException
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.models.evaluation import EvaluationTask
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.evaluation import evaluation_task_repository
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

    def _check_evaluation_task_status(self, db: Session, model_id: str, dataset_id: str, confidence_score: float):
        evaluation_task = evaluation_task_repository.get_by_model_dataset_and_confidence(
            db=db,
            model_id=model_id,
            dataset_id=dataset_id,
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

        conversion_task = self._find_existing_conversion_task(
            db=db,
            input_model_id=evaluation_in.input_model_id,
            target_framework=evaluation_in.framework,
            target_device_name=evaluation_in.device_name,
            target_software_version=evaluation_in.software_version,
            target_data_type=evaluation_in.precision
        )

        try:
            for confidence_score in confidence_scores:
                self._check_evaluation_task_status(db=db, model_id=conversion_task.model_id, dataset_id=evaluation_in.dataset_id, confidence_score=confidence_score)
        except EvaluationTaskAlreadyExistsException:
            raise

        task_result = run_multiple_evaluations.apply_async(
            kwargs={
                "api_key": api_key,
                "model_id": conversion_task.model_id,
                "dataset_id": evaluation_in.dataset_id,
                "training_task_id": evaluation_in.training_task_id,
                "confidence_scores": confidence_scores,
            },
        )

        evaluation_task_id = task_result.get(timeout=5)

        logger.info(f"Evaluation task ID: {evaluation_task_id}")

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
    ) -> List[str]:
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

        evaluation_tasks = evaluator.get_evaluation_results_by_model_and_dataset(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=model_id,
            dataset_id=dataset_id
        )

        return evaluation_tasks

    def get_evaluation_result_details(
        self,
        db: Session,
        api_key: str,
        model_id: str,
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
        evaluation_tasks = evaluator.get_evaluation_results_by_model_and_dataset(
            db=db,
            user_id=netspresso.user_info.user_id,
            model_id=model_id,
            dataset_id=dataset_id
        )

        if not evaluation_tasks:
            logger.warning(f"No evaluation tasks found for model {model_id} and dataset {dataset_id}")
            return EvaluationResultsPayload(
                task_id="",
                dataset_id=dataset_id,
                results=[]
            )

        # Find the most recent completed evaluation task
        completed_tasks = [task for task in evaluation_tasks if task.status == Status.COMPLETED]
        if not completed_tasks:
            logger.warning(f"No completed evaluation tasks found for model {model_id} and dataset {dataset_id}")
            return EvaluationResultsPayload(
                task_id="",
                dataset_id=dataset_id,
                results=[]
            )

        # Get user_id and task_id from the first completed task
        user_id = completed_tasks[0].user_id
        task_id = completed_tasks[0].task_id

        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="evaluation_results_")
        temp_path = Path(temp_dir)

        try:
            # 1. Download predictions.json
            predictions_object_path = f"{user_id}/{task_id}/predictions.json"
            predictions_local_path = temp_path / "predictions.json"

            storage_handler.download_file_from_s3(
                bucket_name=EVALUATION_BUCKET_NAME,
                object_path=predictions_object_path,
                local_path=predictions_local_path.as_posix()
            )

            # 2. Read predictions.json
            predictions_data = FileHandler.load_json(predictions_local_path)

            # 3. Get list of image files from result_images directory
            result_images_prefix = f"{user_id}/{task_id}/result_images/"
            image_objects = storage_handler.list_objects(
                bucket_name=EVALUATION_BUCKET_NAME,
                prefix=result_images_prefix
            )

            # Filter only files with '_images' in the path
            image_paths = [obj for obj in image_objects if '_images' in Path(obj).name]

            # 4. Create presigned URLs for each image
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

            # 5. Combine predictions with image URLs
            all_image_predictions = []

            # Extract base predictions from the loaded file
            if "predictions" in predictions_data:
                base_predictions = predictions_data["predictions"]

                # Sort image paths to ensure consistent ordering (optional)
                image_paths.sort()

                # Limit to the minimum length of both arrays to ensure 1:1 mapping
                max_items = min(len(base_predictions), len(image_paths))

                # Combine predictions with image URLs based on index order
                for i in range(max_items):
                    pred = base_predictions[i]
                    image_path = image_paths[i]

                    # Extract filename from path
                    image_filename = Path(image_path).name

                    image_url = image_urls.get(image_filename)

                    if not image_url:
                        logger.warning(f"No presigned URL found for image {image_filename}")
                        continue

                    # Create prediction entries for each threshold (0.3, 0.5, 0.6)
                    all_bboxes = pred.get("bboxes", [])

                    # For each threshold, filter bboxes
                    threshold_predictions = []
                    for threshold in [0.3, 0.5, 0.6]:
                        filtered_bboxes = [
                            bbox for bbox in all_bboxes
                            if bbox.get("confidence_score", 0) >= threshold
                        ]

                        threshold_predictions.append({
                            "threshold": threshold,
                            "bboxes": filtered_bboxes
                        })

                    # Add to results
                    all_image_predictions.append({
                        "image_id": image_filename,
                        "image_url": image_url,
                        "predictions": threshold_predictions
                    })

            # Apply pagination to image predictions
            total_count = len(all_image_predictions)

            # Validate start index
            if start >= total_count:
                start = 0

            # Calculate end index
            end = min(start + size, total_count)

            # Get paginated results
            paginated_predictions = all_image_predictions[start:end]

            # Return combined results with pagination info
            return EvaluationResultsPayload(
                model_id=model_id,
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

evaluation_task_service = EvaluationTaskService()
