from typing import Union, Optional, Dict

from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.v1.schemas import UserInfo
from netspresso.enums import Framework, DeviceName, DataType, SoftwareVersion
from netspresso.modules.converter.converter_v1 import ConverterV1


class Converter:
    def __init__(self, token_handler: TokenHandler, user_info: UserInfo):
        self.converter = self.__get_converter(
            token_handler=token_handler, user_info=user_info
        )

    def __get_converter(self, token_handler: TokenHandler, user_info: UserInfo):
        # return proper client version by env (cloud : v1, on-prem : v2)
        if True:
            return ConverterV1(token_handler=token_handler, user_info=user_info)
        else:
            return ConverterV1(token_handler=token_handler, user_info=user_info)

    def convert_model(
        self,
        input_model_path: str,
        output_dir: str,
        target_framework: Union[str, Framework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_shape: Optional[InputShape] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
    ) -> Dict:
        pass

    def get_conversion_task(
            self, conversion_task: Union[str, ConversionTask]
    ) -> ConversionTask:
        pass