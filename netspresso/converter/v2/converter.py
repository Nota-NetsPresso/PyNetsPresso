from datetime import time
from typing import Union, Optional, Dict

from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.enums import TaskStatus
from netspresso.clients.launcher.v2.schemas import InputLayer, ResponseConvertTaskItem
from netspresso.enums import Framework, DeviceName, DataType, SoftwareVersion, TaskType
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class ConverterV2:
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Converter."""

        self.token_handler = token_handler
        self.user_info = user_info

    def convert_model(
        self,
        input_model_path: str,
        output_dir: str,
        target_framework: Union[str, Framework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_layer: Optional[InputLayer] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
    ) -> Dict:
        # 1. get presigned url
        # 2. upload model file
        # 3. validate model file
        # 4. start task

        # FileHandler.check_input_model_path(input_model_path)
        #
        # self.token_handler.validate_token()
        #
        # FileHandler.create_unique_folder(folder_path=output_dir)
        # metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.CONVERT)



        # GET presigned_model_upload_url
        presigned_url_response = (
            launcher_client_v2.converter.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )
        )

        # UPLOAD model_file
        model_upload_response = launcher_client_v2.converter.upload_model_file(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=input_model_path,
            presigned_upload_url=presigned_url_response.data.presigned_upload_url,
        )

        # VALIDATE model_file
        validate_model_response = launcher_client_v2.converter.validate_model_file(
            access_token=self.token_handler.tokens.access_token,
            input_model_path=input_model_path,
            ai_model_id=presigned_url_response.data.ai_model_id,
        )

        # START convert task
        response = launcher_client_v2.converter.start_task(
            access_token=self.token_handler.tokens.access_token,
            input_model_id=presigned_url_response.data.ai_model_id,
            target_device_name=target_device_name,
            target_framework=target_framework,
            data_type=target_data_type,
            input_layer=input_layer,
            software_version=target_software_version,
        )

        if wait_until_done:
            while True:
                # Poll Convert Task status
                response = launcher_client_v2.converter.read_task(
                    access_token=self.token_handler.tokens.access_token,
                    task_id=response.data.convert_task_id,
                )
                if response.data.status in [
                    TaskStatus.FINISHED.value,
                    TaskStatus.CANCELLED.value,
                ]:
                    break
                time.sleep(3)

        # metadata.update_converted_model_path(
        #     converted_model_path=default_model_path.with_suffix(extension).as_posix()
        # )
        # metadata.update_model_info(
        #     data_type=model.data_type,
        #     framework=model.framework,
        #     input_shape=model.input_shape,
        # )
        # metadata.update_convert_info(
        #     target_framework=conversion_task.target_framework,
        #     target_device_name=conversion_task.target_device_name,
        #     data_type=conversion_task.data_type,
        #     software_version=conversion_task.software_version,
        #     model_file_name=conversion_task.model_file_name,
        #     convert_task_uuid=conversion_task.convert_task_uuid,
        #     input_model_uuid=conversion_task.input_model_uuid,
        #     output_model_uuid=conversion_task.output_model_uuid,
        # )
        # metadata.update_status(status=Status.COMPLETED)
        # metadata.update_available_devices(converter_uploaded_model.available_devices)
        # MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

        return response

    def get_conversion_task(self, conversion_task_id: str) -> ResponseConvertTaskItem:
        pass
