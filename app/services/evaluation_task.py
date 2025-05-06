from typing import Any, Dict, List, Optional

from fastapi import HTTPException
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
from app.api.v1.schemas.task.evaluation.evaluation_task import EvaluationCreate
from app.services.training_task import train_task_service
from app.worker.evaluation_task import poll_evaluation_status, run_multiple_evaluations
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums import DataType, DeviceName, SoftwareVersion, Status
from netspresso.enums.conversion import SourceFramework, TargetFramework
from netspresso.exceptions.trainer import NotCompletedTrainingException
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.evaluation import evaluation_result_repository, evaluation_task_repository
from netspresso.utils.db.session import get_db_session

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"
POLLING_INTERVAL = 30  # seconds


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
        with get_db_session() as db:
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

        return None

    def create_evaluation_task(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        # 1. 학습 태스크가 완료되었는지 확인
        training_task = train_task_service.get_training_task(db=db, task_id=evaluation_in.training_task_id, api_key=api_key)
        if training_task.status != Status.COMPLETED:
            raise NotCompletedTrainingException(training_task_id=evaluation_in.training_task_id)

        # 2. 변환 태스크가 이미 존재하는지 확인
        conversion_task = self._find_existing_conversion_task(
            input_model_id=evaluation_in.input_model_id,
            target_framework=evaluation_in.framework,
            target_device_name=evaluation_in.device_name,
            target_software_version=evaluation_in.software_version,
            target_data_type=evaluation_in.precision
        )

        # 3. 변환 태스크가 없으면 새로 생성
        if conversion_task is None:
            # 변환기 인스턴스 가져오기
            netspresso = NetsPresso(api_key=api_key)
            converter = netspresso.converter_v2()

            # 변환 프로세스 시작
            conversion_task_id = converter.convert_model_from_id(
                input_model_id=evaluation_in.input_model_id,
                target_framework=evaluation_in.framework,
                target_device_name=evaluation_in.device_name,
                target_software_version=evaluation_in.software_version,
                target_data_type=evaluation_in.precision,
                wait_until_done=True  # 변환이 완료될 때까지 대기
            )

            # 변환 태스크 결과 가져오기
            conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=conversion_task_id)

        # 4. Celery 태스크를 시작하여 여러 신뢰도 점수에 대한 평가 수행
        task_result = run_multiple_evaluations.delay(
            api_key=api_key,
            model_id=conversion_task.model_id,
            dataset_id=evaluation_in.dataset_id,
            training_task_id=evaluation_in.training_task_id,
            conversion_task_id=conversion_task.task_id,
        )

        # 5. 평가 태스크 상태 폴링 시작
        evaluation_task_id = task_result.get(timeout=5)  # 평가 태스크 ID 가져오기 (5초 타임아웃)
        poll_evaluation_status.apply_async(
            args=[api_key, evaluation_task_id],
            countdown=POLLING_INTERVAL
        )

        # 평가 태스크 ID 반환
        return evaluation_task_id

    def get_evaluation_task_with_results(
        self,
        db: Session,
        task_id: str,
        api_key: str = None
    ) -> Dict[str, Any]:
        """
        평가 태스크와 연관된 모든 confidence_score별 결과를 조회합니다.

        Args:
            db: 데이터베이스 세션
            task_id: 평가 태스크 ID
            api_key: 인증에 사용할 API 키 (선택적)

        Returns:
            평가 태스크와 결과 정보가 포함된 응답
        """
        # 평가 태스크 조회
        task = evaluation_task_repository.get_by_task_id(db=db, task_id=task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"평가 태스크 ID {task_id}를 찾을 수 없습니다.")

        # 연관된 평가 결과 조회
        results = evaluation_result_repository.get_by_evaluation_task_id(db=db, evaluation_task_id=task_id)

        # 결과 데이터 변환
        result_data = []
        for result in results:
            result_data.append({
                "result_id": result.result_id,
                "confidence_score": result.confidence_score,
                "metric_unit": result.metric_unit,
                "metric_value": result.metric_value,
                "results_path": result.results_path,
                "status": result.status,
                "error_detail": result.error_detail
            })

        # 데이터셋 정보 가져오기
        dataset_name = ""
        is_dataset_deleted = False

        # TODO: 필요시 데이터셋 정보 조회 로직 추가

        # 응답 데이터 구성
        return {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": dataset_name,
            "is_dataset_deleted": is_dataset_deleted,
            "input_model_id": task.input_model_id,
            "training_task_id": task.training_task_id,
            "conversion_task_id": task.conversion_task_id,
            "status": task.status,
            "error_detail": task.error_detail,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "results": result_data
        }


evaluation_task_service = EvaluationTaskService()
