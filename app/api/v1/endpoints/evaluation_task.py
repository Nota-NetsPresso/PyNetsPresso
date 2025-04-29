import copy
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.device import SupportedDevicesResponse
from app.api.v1.schemas.task.evaluation.evaluation_task import (
    EvaluationCreate,
    EvaluationCreatePayload,
    EvaluationCreateResponse,
    EvaluationPayload,
    EvaluationResponse,
    EvaluationResultsPayload,
    EvaluationResultsResponse,
)
from app.services.evaluation_task import evaluation_task_service
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.enums.conversion import SourceFramework
from netspresso.utils.db.session import get_db

router = APIRouter()
storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"  # S3 버킷 이름


@router.get(
    "/evaluations/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices and frameworks for model evaluation based on the source framework.",
)
def get_supported_evaluation_devices(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    framework = SourceFramework.ONNX
    supported_devices = evaluation_task_service.get_supported_devices(db=db, framework=framework, api_key=api_key)

    return SupportedDevicesResponse(data=supported_devices)


@router.post("/evaluations", response_model=EvaluationCreateResponse, status_code=201)
def create_evaluations_task(
    request_body: EvaluationCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationCreateResponse:
    # evaluation_task = evaluation_task_service.create_evaluation_task(db=db, evaluation_in=request_body, api_key=api_key)
    evaluation_task = EvaluationCreatePayload(task_id="task_uFUOSIObHX")

    return EvaluationCreateResponse(data=evaluation_task)


@router.get("/evaluations/{task_id}", response_model=EvaluationResponse, status_code=200)
def get_evaluation_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationResponse:
    # evaluation_task = evaluation_task_service.get_evaluation_task(db=db, task_id=task_id, api_key=api_key)
    evaluation_task = EvaluationPayload(
        task_id="task_uFUOSIObHX",
        dataset_id="dataset_uFUOSIObHX",
        dataset_name="project1_traffic-sign (#1000)",
        is_dataset_deleted=False,
        metric_unit="mAP@[.50]",
        metric_value=0.6,
        results_path="",
        input_model_id="model_ZrvXDmbFjq",
        training_task_id="task_VpbB5OiQmL",
        conversion_task_id="task_bH_MroNkjO",
        status="completed",
        is_deleted=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    return EvaluationResponse(data=evaluation_task)


@router.get("/evaluations/{task_id}/results", response_model=EvaluationResultsResponse, status_code=200)
def get_evaluation_results(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationResultsResponse:
    try:
        # 1. 이미지 파일을 숫자 순서대로 정렬
        image_dir = "assets/images"

        if not os.path.exists(image_dir):
            # 개발/테스트 환경에서 이미지 디렉토리가 없을 경우 더미 URL 생성
            image_urls = [f"https://storage.example.com/netspresso/images/dummy_{i}.png" for i in range(10)]
        else:
            # 이미지 파일을 숫자 순서대로 정렬
            image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

            # 숫자 추출 함수
            def extract_number(filename):
                match = re.search(r'(\d+)_images\.png', filename)
                return int(match.group(1)) if match else 999999

            # 숫자 순서대로 정렬
            image_files.sort(key=extract_number)

            # 2. 이미지 파일을 S3에 업로드하고 presigned URL 생성
            image_urls = []
            task_prefix = f"evaluation_tasks/{task_id}/images"

            for img_file in image_files:
                # 이미지 파일 경로
                local_path = os.path.join(image_dir, img_file)

                # S3에 저장될 객체 경로 생성
                object_path = f"{task_prefix}/{img_file}"

                try:
                    # 파일이 이미 S3에 존재하는지 확인
                    if storage_handler.check_file_exists(bucket_name=BUCKET_NAME, object_path=object_path):
                        # 다운로드용 presigned URL 생성
                        presigned_url = storage_handler.get_download_presigned_url(
                            bucket_name=BUCKET_NAME,
                            object_path=object_path,
                            download_name=img_file,
                            expires_in=86400  # 24시간 유효
                        )
                        image_urls.append(presigned_url)
                    else:
                        # 파일이 없는 경우 로깅하고 로컬 이미지 경로로 더미 URL 생성
                        print(f"File does not exist in S3: {object_path}")
                        image_urls.append(f"https://storage.example.com/netspresso/images/{img_file}")
                except Exception as e:
                    # URL 생성 실패 시 로깅하고 더미 URL 추가
                    print(f"Error generating presigned URL for {img_file}: {str(e)}")
                    image_urls.append(f"https://storage.example.com/netspresso/images/{img_file}")

        # 3. predictions.json 파일 읽기
        predictions_file = Path("assets/predictions.json")
        results = []

        if not predictions_file.exists():
            # 개발/테스트 환경에서 파일이 없을 경우 빈 예측 리스트 생성
            for i, image_url in enumerate(image_urls):
                # 이미지 ID 추출 (파일명 또는 인덱스 기반)
                image_id = image_files[i] if i < len(image_files) else f"dummy_{i}"

                # 여러 threshold에 대한 빈 예측 생성
                predictions = []
                for threshold in [0.3, 0.5, 0.6]:
                    predictions.append({
                        "threshold": threshold,
                        "bboxes": []
                    })

                # 이미지 예측 결과 추가
                results.append({
                    "image_id": image_id,
                    "image_url": image_url,
                    "predictions": predictions
                })
        else:
            with open(predictions_file, "r") as f:
                predictions_data = json.load(f)

            # 기본 predictions 리스트 가져오기
            base_predictions = predictions_data.get("predictions", [])

            # 이미지 파일 수만큼 predictions 잘라내기 (만약 이미지보다 predictions가 더 많다면)
            base_predictions = base_predictions[:len(image_urls)]

            # 이미지 파일보다 predictions가 적다면 빈 예측으로 채우기
            while len(base_predictions) < len(image_urls):
                base_predictions.append({"bboxes": []})

            # 각 이미지에 대한 결과 생성
            for i, (image_url, pred) in enumerate(zip(image_urls, base_predictions)):
                # 이미지 ID 추출 (파일명 또는 인덱스 기반)
                image_id = image_files[i] if i < len(image_files) else f"dummy_{i}"

                # 여러 threshold에 대한 예측 결과 생성
                predictions = []
                for threshold in [0.3, 0.5, 0.6]:
                    # 각 이미지의 예측 결과에서 confidence threshold 이상인 bboxes만 필터링
                    filtered_bboxes = [
                        bbox for bbox in pred.get("bboxes", [])
                        if bbox.get("confidence_score", 0) >= threshold
                    ]

                    # 해당 threshold의 예측 결과 추가
                    predictions.append({
                        "threshold": threshold,
                        "bboxes": filtered_bboxes
                    })

                # 이미지 예측 결과 추가
                results.append({
                    "image_id": image_id,
                    "image_url": image_url,
                    "predictions": predictions
                })

        # 4. 응답 구성
        evaluation_results = EvaluationResultsPayload(
            task_id=task_id,
            dataset_id="dataset_uFUOSIObHX",  # 실제 구현에서는 DB에서 가져온 dataset_id 사용
            results=results,
        )

        # 결과를 200개로 확장
        expanded_results = []
        original_count = len(results)

        # 기존 결과 추가
        expanded_results.extend(results)

        # 나머지를 채우기 위해 기존 결과를 반복해서 추가
        if original_count < 200:
            needed_copies = (200 - original_count) // original_count + 1

            for copy_num in range(needed_copies):
                for i, result in enumerate(results):
                    if len(expanded_results) >= 200:
                        break

                    # 원본 결과의 복사본 생성 (깊은 복사)
                    copied_result = copy.deepcopy(result)

                    # 고유한 image_id 생성
                    copied_result["image_id"] = f"{result['image_id']}_copy_{copy_num+1}_{i}"

                    expanded_results.append(copied_result)

        # 정확히 200개로 자르기
        expanded_results = expanded_results[:200]

        # 확장된 결과로 업데이트
        evaluation_results.results = expanded_results

        return EvaluationResultsResponse(data=evaluation_results)

    except Exception as e:
        # 실제 구현에서는 로깅 등의 처리가 필요
        raise HTTPException(status_code=500, detail=f"Error generating evaluation results: {str(e)}")
