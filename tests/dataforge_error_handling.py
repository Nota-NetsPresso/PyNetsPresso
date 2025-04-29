# test_error_handling.py
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from netspresso.trainer.storage.dataforge import Split, dataforge

# 테스트용 UUID 입력 (실제 존재하는 데이터셋 UUID 필요)
dataset_uuid = "32d71060-786e-4020-96b1-b0f08c0f70e6"
output_dir = "./test_error_download"

def setup_test():
    """테스트 환경 설정"""
    # 이전 테스트 데이터가 있다면 삭제
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 테스트 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)

    # 데이터셋 버전 정보 가져오기
    logger.info(f"Getting dataset version for UUID: {dataset_uuid}")
    return dataforge.get_latest_dataset_version(dataset_uuid=dataset_uuid, split=Split.TRAIN)

def test_network_error():
    """네트워크 오류 시뮬레이션 테스트"""
    dataset_version = setup_test()
    if not dataset_version:
        logger.error("Failed to get dataset version")
        return False

    # 첫 번째 단계: 일부 파일 정상 다운로드
    logger.info("Step 1: Downloading some files normally")

    # 다운로드 시작 후 몇 개의 파일만 처리하도록 패치
    original_download_files = dataforge._download_data_files
    download_count = [0]  # 가변 객체로 카운터 생성

    def mock_download_files(row, dataset_dir):
        download_count[0] += 1
        # 3개의 파일만 정상 다운로드 후 네트워크 오류 발생
        if download_count[0] <= 3:
            return original_download_files(row, dataset_dir)
        else:
            # 네트워크 오류 시뮬레이션
            raise ConnectionError("Simulated network error")

    # 다운로드 함수 패치
    with patch.object(dataforge, '_download_data_files', side_effect=mock_download_files):
        try:
            result = dataforge.download_dataset(dataset_version=dataset_version, output_dir=output_dir)
            logger.info("First download attempt completed unexpectedly")
        except Exception as e:
            logger.warning(f"Expected error in first attempt: {e}")

    # 원래 함수 복원
    dataforge._download_data_files = original_download_files

    # 두 번째 단계: 이어받기 시도
    logger.info("Step 2: Resuming download after network error")
    result = dataforge.download_dataset(dataset_version=dataset_version, output_dir=output_dir)

    if result:
        logger.success("Resume after network error test passed")
        return True
    else:
        logger.error("Resume after network error test failed")
        return False

def test_corrupt_status_file():
    """손상된 상태 파일 테스트"""
    dataset_version = setup_test()
    if not dataset_version:
        logger.error("Failed to get dataset version")
        return False

    # 첫 번째 단계: 일부 파일 정상 다운로드
    logger.info("Step 1: Starting initial download")

    # 몇 개 파일만 다운로드하고 중단
    original_download_files = dataforge._download_data_files
    download_count = [0]

    def mock_download_files_limited(row, dataset_dir):
        download_count[0] += 1
        if download_count[0] <= 5:
            return original_download_files(row, dataset_dir)
        else:
            # 5개 파일 후 중단
            logger.info("Downloaded 5 files, stopping first attempt")
            return True, True  # 성공으로 처리하고 중단

    with patch.object(dataforge, '_download_data_files', side_effect=mock_download_files_limited):
        dataforge.download_dataset(dataset_version=dataset_version, output_dir=output_dir)

    # 원래 함수 복원
    dataforge._download_data_files = original_download_files

    # 상태 파일 찾기
    dataset_dir = Path(output_dir) / dataset_version.data.dataset_uuid
    csv_path = dataset_version.data.dataset_metadata.csv_s3_path
    status_csv_path = dataset_dir / f"{Path(csv_path).stem}_status.csv"

    if status_csv_path.exists():
        # 상태 파일 손상시키기
        logger.info("Step 2: Corrupting status file")
        with open(status_csv_path, 'w') as f:
            f.write("Corrupted content that is not valid CSV")

        # 이어받기 시도
        logger.info("Step 3: Attempting to resume with corrupted status file")
        result = dataforge.download_dataset(dataset_version=dataset_version, output_dir=output_dir)

        if result:
            logger.success("Resume after corrupt status file test passed")
            return True
        else:
            logger.error("Resume after corrupt status file test failed")
            return False
    else:
        logger.error("Status file not found, test failed")
        return False

def test_disk_full_error():
    """디스크 공간 부족 오류 시뮬레이션"""
    dataset_version = setup_test()
    if not dataset_version:
        logger.error("Failed to get dataset version")
        return False

    # 첫 번째 단계: 일부 파일 정상 다운로드
    logger.info("Step 1: Downloading some files normally")

    # 다운로드 시작 후 몇 개의 파일만 처리하도록 패치
    original_update_status = dataforge._update_status_file
    file_count = [0]

    def mock_disk_full_after_few_files(status_csv_path, rows):
        file_count[0] += 1
        # 4개의 파일은 정상 처리 후 디스크 공간 부족 시뮬레이션
        if file_count[0] <= 4:
            return original_update_status(status_csv_path, rows)
        else:
            # 디스크 용량 부족 오류 시뮬레이션
            raise OSError("Simulated disk full error")

    # 상태 파일 업데이트 함수 패치
    with patch.object(dataforge, '_update_status_file', side_effect=mock_disk_full_after_few_files):
        try:
            result = dataforge.download_dataset(dataset_version=dataset_version, output_dir=output_dir)
            logger.info("First download attempt completed unexpectedly")
        except Exception as e:
            logger.warning(f"Expected error in first attempt: {e}")

    # 원래 함수 복원
    dataforge._update_status_file = original_update_status

    # 두 번째 단계: 이어받기 시도
    logger.info("Step 2: Resuming download after disk full error")
    result = dataforge.download_dataset(dataset_version=dataset_version, output_dir=output_dir)

    if result:
        logger.success("Resume after disk full error test passed")
        return True
    else:
        logger.error("Resume after disk full error test failed")
        return False

def run_all_tests():
    """모든 테스트 실행"""
    test_results = {}

    # 네트워크 오류 테스트
    logger.info("=== Testing resume after network error ===")
    test_results["network_error"] = test_network_error()

    # 다음 테스트를 위해 디렉토리 초기화
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    time.sleep(1)

    # 손상된 상태 파일 테스트
    logger.info("\n=== Testing resume with corrupted status file ===")
    test_results["corrupt_status_file"] = test_corrupt_status_file()

    # 다음 테스트를 위해 디렉토리 초기화
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    time.sleep(1)

    # 디스크 공간 부족 테스트
    logger.info("\n=== Testing resume after disk full error ===")
    test_results["disk_full_error"] = test_disk_full_error()

    # 결과 요약
    logger.info("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        logger.success("All tests passed successfully!")
    else:
        logger.error("Some tests failed. See above for details.")

if __name__ == "__main__":
    run_all_tests()
