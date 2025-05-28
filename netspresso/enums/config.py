from netspresso.enums.base import StrEnum


class EnvironmentType(StrEnum):
    V2_PROD_CLOUD = "v2-prod-cloud"
    V2_PROD_ON_PREM = "v2-prod-on-prem"
    V2_STAGING_CLOUD = "v2-staging-cloud"
    V2_STAGING_ON_PREM = "v2-staging-on-prem"
    V2_DEV_CLOUD = "v2-dev-cloud"
    V2_DEV_ON_PREM = "v2-dev-on-prem"


class ServiceName(StrEnum):
    NP = "NP"


class ServiceModule(StrEnum):
    AUTH = "AUTH"
    COMPRESSOR = "COMPRESSOR"
    LAUNCHER = "LAUNCHER"
    TRAINER = "TRAINER"


class EndPointProperty(StrEnum):
    HOST = "HOST"
    PORT = "PORT"
    URI_PREFIX = "URI_PREFIX"
