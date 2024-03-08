from loguru import logger

from netspresso.clients.config import Config, Module
from netspresso.clients.tao.auth import AuthAPI
from netspresso.clients.tao.dataset import DatasetAPI
from netspresso.clients.tao.experiment import ExperimentAPI
from netspresso.clients.utils.common import create_tao_headers


class TAOAPIClient:
    def __init__(self):
        self.config = Config(Module.TAO)
        self.host = self.config.HOST
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}{self.prefix}"
        self.auth = self._create_auth_api()
        self.dataset = self._create_dataset_api()
        self.experiment = self._create_experiment_api()

    def _create_auth_api(self):
        return AuthAPI(self.url)

    def _create_dataset_api(self):
        return DatasetAPI(self.url)

    def _create_experiment_api(self):
        return ExperimentAPI(self.url)


class TAOTokenHandler:
    def __init__(self, ngc_api_key: str) -> None:
        try:
            data = {"ngc_api_key": ngc_api_key}
            credentials = tao_client.auth.login(data)
            self.user_id = credentials["user_id"]
            self.token = credentials["token"]
            self.headers = create_tao_headers(self.token)
            logger.info("Login was successfully to TAO.")

        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e


tao_client = TAOAPIClient()
