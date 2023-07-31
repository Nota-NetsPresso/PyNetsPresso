import os, configparser
from pathlib import Path
from enum import Enum
from typing import Literal
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
config = configparser.ConfigParser()
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "PROD")
logger.info(f"Read {DEPLOYMENT_MODE} config")
config.read(f"{BASE_DIR}/configs/config.ini")

class EnvironmentType(str, Enum):
    PROD = "PROD"
    LOCAL = "LOCAL"

    @classmethod
    def create_literal(cls):
        return Literal[
            "PROD",
            "LOCAL"
        ]

class EndPoint(str, Enum):
    GENRAL = "GENERAL"
    COMPRESSOR = "COMPRESSOR"
    LAUNCHX = "LAUNCHX"

    @classmethod
    def create_literal(cls):
        return Literal[
            "GENERAL",
            "COMPRESSOR",
            "LAUNCHX"
        ]
    
class EndPointProperty(str, Enum):
    HOST = "HOST"
    PORT = "PORT"
    URI_PREFIX = "URI_PREFIX"

    @classmethod
    def create_literal(cls):
        return Literal[
            "HOST",
            "PORT",
            "URI_PREFIX"
        ]
    

class Config:
    ENVIRONMENT_TYPE: EnvironmentType = EnvironmentType.PROD
    CONFIG_SESSION: str = f"{EndPoint.GENRAL}.{ENVIRONMENT_TYPE}"
    HOST: str = config[CONFIG_SESSION][EndPointProperty.HOST]
    PORT: int = int(config[CONFIG_SESSION][EndPointProperty.PORT])
    URI_PREFIX: str = config[CONFIG_SESSION][EndPointProperty.URI_PREFIX]

    def __init__(self, endpoint: EndPoint = EndPoint.GENRAL):
        self.ENVIRONMENT_TYPE = EnvironmentType(DEPLOYMENT_MODE.upper())
        self.CONFIG_SESSION = f"{endpoint}.{self.ENVIRONMENT_TYPE}"
        self.HOST = config[self.CONFIG_SESSION][EndPointProperty.HOST]
        self.PORT = int(config[self.CONFIG_SESSION][EndPointProperty.PORT])
        self.URI_PREFIX = config[self.CONFIG_SESSION][EndPointProperty.URI_PREFIX]

