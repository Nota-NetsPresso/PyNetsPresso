from netspresso.clients.config import Config, ServiceModule, ServiceName
from netspresso.clients.launcher.v2.benchmarker import Benchmarker
from netspresso.clients.launcher.v2.converter import Converter
from netspresso.clients.launcher.v2.graph_optimizer import GraphOptimizer
from netspresso.clients.launcher.v2.quantizer import Quantizer
from netspresso.clients.launcher.v2.simulator import Simulator


class LauncherAPIClient:
    def __init__(self):
        self.config = Config(ServiceName.NP, ServiceModule.LAUNCHER)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}:{self.port}{self.prefix}"
        self.converter = self._create_convert_api()
        self.benchmarker = self._create_benchmark_api()
        self.quantizer = self._create_quantize_api()
        self.graph_optimizer = self._create_graph_optimizer_api()
        self.simulator = self._create_simulator_api()

    def _create_convert_api(self):
        return Converter(self.url)

    def _create_benchmark_api(self):
        return Benchmarker(self.url)

    def _create_quantize_api(self):
        return Quantizer(self.url)

    def _create_graph_optimizer_api(self):
        return GraphOptimizer(self.url)

    def _create_simulator_api(self):
        return Simulator(self.url)

    def is_cloud(self) -> bool:
        return self.config.is_cloud()


launcher_client_v2 = LauncherAPIClient()
