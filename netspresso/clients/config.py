import configparser
import os
from enum import Enum
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
config_parser = configparser.ConfigParser()
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "PROD")
logger.info(f"Read {DEPLOYMENT_MODE} config")
config_parser.read(f"{BASE_DIR}/configs/config-{DEPLOYMENT_MODE.lower()}.ini")


class EnvironmentType(str, Enum):
    PROD = "prod"
    LOCAL = "local"


class Module(str, Enum):
    GENERAL = "GENERAL"
    COMPRESSOR = "COMPRESSOR"
    LAUNCHER = "LAUNCHER"
    TAO = "TAO"


class EndPointProperty(str, Enum):
    HOST = "HOST"
    PORT = "PORT"
    URI_PREFIX = "URI_PREFIX"


class Config:
    def __init__(self, module: Module = Module.GENERAL):
        self.ENVIRONMENT_TYPE = EnvironmentType(DEPLOYMENT_MODE.lower())
        self.MODULE = module
        self.HOST = config_parser[self.MODULE][EndPointProperty.HOST]
        self.PORT = int(config_parser[self.MODULE][EndPointProperty.PORT])
        self.URI_PREFIX = config_parser[self.MODULE][EndPointProperty.URI_PREFIX]
