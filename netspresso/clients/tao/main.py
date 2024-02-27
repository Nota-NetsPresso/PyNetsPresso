from netspresso.clients.config import Config, Module
from netspresso.clients.tao.auth import AuthAPI
from netspresso.clients.tao.dataset import DatasetAPI
from netspresso.clients.tao.experiment import ExperimentAPI


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


tao_client = TAOAPIClient()
