import os
import threading
import time

import pika
from celery import Celery
from celery.signals import worker_process_init
from loguru import logger

# 로그 디렉토리
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# loguru 로깅 설정
#logger.remove()  # 기본 핸들러 제거
#logger.add(
#    os.path.join(LOGS_DIR, 'celery.log'),
#    rotation="10 MB",
#    level="INFO",
#    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
#)

# 환경 변수에서 설정 값 가져오기
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'amqp://guest:guest@rabbitmq:5672//')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'rpc://')
CELERY_WORKER_PREFETCH_MULTIPLIER = int(os.environ.get('CELERY_WORKER_PREFETCH_MULTIPLIER', '1'))
CELERY_WORKER_MAX_TASKS_PER_CHILD = int(os.environ.get('CELERY_WORKER_MAX_TASKS_PER_CHILD', '1'))

# Celery 설정
celery_app = Celery("netspresso")
celery_app.conf.update(
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    worker_prefetch_multiplier=CELERY_WORKER_PREFETCH_MULTIPLIER,
    worker_max_tasks_per_child=CELERY_WORKER_MAX_TASKS_PER_CHILD,
    broker_pool_limit=None,
    task_acks_late=True,  # 작업 시작 시 ack 보내도록 변경
    broker_heartbeat=None,
    task_track_started=True,
    include=[
        "app.worker.training_task",
        "app.worker.conversion_task",
        "app.worker.benchmark_task",
        "app.worker.evaluation_task",
        "app.worker.compression_task",
    ],
    result_expires=86400,  # one day,
)

# worker_process_init 함수 내에 추가
@worker_process_init.connect
def setup_worker(sender=None, **kwargs):
    """워커 프로세스 초기화 시 호출됨"""
    logger.info("Initializing worker process")
