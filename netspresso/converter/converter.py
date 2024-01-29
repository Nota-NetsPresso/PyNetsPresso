import time
from pathlib import Path
from typing import Dict, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import auth_client, TokenHandler
from netspresso.clients.launcher import launcher_client
from netspresso.clients.launcher.schemas import TargetDeviceFilter
from netspresso.clients.launcher.schemas.model import ConversionTask, InputShape, Model
from netspresso.enums import (
    DataType,
    DeviceName,
    Framework,
    Module,
    ServiceCredit,
    SoftwareVersion,
    Status,
    TaskStatus,
    TaskType,
)

from ..utils import FileHandler, check_credit_balance
from ..utils.metadata import MetadataHandler


class Converter:
    def __init__(self, token_handler: TokenHandler, user_info):
        """Initialize the Model Converter."""

        self.token_handler = token_handler
        self.user_info = user_info

    def convert_model(
        self,
        input_model_path: Union[Path, str],
        output_dir: Union[Path, str],
        target_framework: Union[str, Framework],
        target_data_type: DataType = DataType.FP16,
        target_device_name: DeviceName = None,
        target_software_version: SoftwareVersion = None,
        input_shape: InputShape = None,
        dataset_path: str = None,
        wait_until_done: bool = True,
    ) -> Dict:
        """Convert a model into the type the specific framework.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the converted model.
            target_framework (Framework | str): the target framework name.
            target_data_type (DataType): data type of the model.
            target_device_name (DeviceName): target device name. Necessary field if target_device is not given.
            target_software_version (SoftwareVersion): target_software_version. Necessary field if target_device_name is one of jetson devices.
            input_shape (InputShape) : target input shape to convert. (ex: dynamic batch to static batch)
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.

        Raises:
            e: If an error occurs while converting the model.

        Returns:
            Dict: model conversion task dict.
        """

        FileHandler.check_input_model_path(input_model_path)
        
        self.token_handler.validate_token()

        try:
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_dir, framework=target_framework
            )
            FileHandler.create_folder(folder_path=output_dir)
            metadata = MetadataHandler.init_metadata(folder_path=output_dir, task_type=TaskType.CONVERT)

            current_credit = auth_client.get_credit(self.token_handler.tokens.access_token)
            check_credit_balance(user_credit=current_credit, service_credit=ServiceCredit.MODEL_CONVERT)
            model = launcher_client.upload_model(
                model_file_path=input_model_path, target_function=Module.CONVERT, access_token=self.token_handler.tokens.access_token
            )

            model_uuid = model
            if type(model) is Model:
                model_uuid = model.model_uuid
                if input_shape is None and model.input_shape is not None:
                    input_shape = model.input_shape
                if target_framework is None and model.framework is not None:
                    target_framework = model.framework

            if target_device_name is None:
                raise NotImplementedError("The conversion is unavailable. Please set target_device_name.")

            elif target_device_name in DeviceName.JETSON_DEVICES and target_software_version is None:
                raise NotImplementedError(
                    "The conversion is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                )

            if type(model) is not Model:
                raise NotImplementedError(
                    "The conversion is unavailable. Please set target_device while using model's uuid string for the conversion."
                )

            # Check available int8 converting devices
            if target_data_type == DataType.INT8:
                if target_device_name not in DeviceName.AVAILABLE_INT8_DEVICES:
                    raise Exception(f"int8 converting supports only {DeviceName.AVAILABLE_INT8_DEVICES}.")
            else:  # FP16, FP32
                if target_device_name in DeviceName.ONLY_INT8_DEVICES:
                    raise Exception(f"{DeviceName.ONLY_INT8_DEVICES} only support int8 data types.")

            devices = TargetDeviceFilter.filter_devices_with_device_name(
                name=target_device_name, devices=model.available_devices
            )

            if target_device_name in DeviceName.JETSON_DEVICES:
                devices = TargetDeviceFilter.filter_devices_with_device_software_version(
                    software_version=target_software_version, devices=devices
                )

            if len(devices) < 1:
                raise NotImplementedError(
                    "The conversion is unavailable. There is no available device with given target_device_name and target_software_version."
                )

            target_device = devices[0]

            logger.info(f"Converting Model for {target_device.device_name} ({target_framework})")

            conversion_task = launcher_client.convert_model(
                user_uuid=self.user_info.user_id,
                model_uuid=model_uuid,
                input_shape=input_shape,
                target_framework=target_framework,
                target_device=target_device.device_name,
                data_type=target_data_type,
                software_version=target_device.software_version,
                dataset_path=dataset_path,
                access_token=self.token_handler.tokens.access_token,
            )

            conversion_task = self.get_conversion_task(conversion_task)

            if wait_until_done:
                while conversion_task.status in [
                    TaskStatus.IN_QUEUE,
                    TaskStatus.IN_PROGRESS,
                ]:
                    conversion_task = self.get_conversion_task(conversion_task)
                    time.sleep(1)

            self.download_converted_model(conversion_task, default_model_path.with_suffix(extension))

            converter_uploaded_model = launcher_client.upload_model(
                model_file_path=default_model_path.with_suffix(extension),
                target_function=Module.BENCHMARK,
                access_token=self.token_handler.tokens.access_token,
            )

            metadata.update_converted_model_path(
                converted_model_path=default_model_path.with_suffix(extension).as_posix()
            )
            metadata.update_model_info(
                data_type=model.data_type,
                framework=model.framework,
                input_shape=model.input_shape,
            )
            metadata.update_convert_info(
                target_framework=conversion_task.target_framework,
                target_device_name=conversion_task.target_device_name,
                data_type=conversion_task.data_type,
                software_version=conversion_task.software_version,
                model_file_name=conversion_task.model_file_name,
                convert_task_uuid=conversion_task.convert_task_uuid,
                input_model_uuid=conversion_task.input_model_uuid,
                output_model_uuid=conversion_task.output_model_uuid,
            )
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_devices(converter_uploaded_model.available_devices)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

            remaining_credit = auth_client.get_credit(self.token_handler.tokens.access_token)
            logger.info(
                f"{ServiceCredit.MODEL_CONVERT} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            return metadata.asdict()

        except Exception as e:
            logger.error(f"Convert failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_dir)

    def get_conversion_task(self, conversion_task: Union[str, ConversionTask]) -> ConversionTask:
        """Get the conversion task information with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """

        self.token_handler.validate_token()

        try:
            conversion_task_uuid = None
            if type(conversion_task) is str:
                conversion_task_uuid = conversion_task
            elif type(conversion_task) is ConversionTask:
                conversion_task_uuid = conversion_task.convert_task_uuid
            else:
                raise NotImplementedError(
                    "There is no available function for the given parameter. The 'conversion_task' should be a UUID string or a ModelConversion object."
                )
            return launcher_client.get_conversion_task(
                conversion_task_uuid=conversion_task_uuid, access_token=self.token_handler.tokens.access_token
            )

        except Exception as e:
            logger.error(f"Get conversion task failed. Error: {e}")
            raise e

    def download_converted_model(self, conversion_task: Union[str, ConversionTask], local_path: str):
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """

        self.token_handler.validate_token()

        try:
            conversion_task_uuid = None
            if type(conversion_task) is str:
                conversion_task_uuid = conversion_task
            elif type(conversion_task) is ConversionTask:
                conversion_task_uuid = conversion_task.convert_task_uuid

            conversion_result: ConversionTask = self.get_conversion_task(conversion_task_uuid)
            if conversion_result.status is TaskStatus.ERROR:
                raise FileNotFoundError("The conversion is Failed. There is no file available for download.")
            if conversion_result.status is not TaskStatus.FINISHED:
                raise FileNotFoundError(
                    "The conversion is in progress. There is no file available for download at the moment."
                )

            download_url = launcher_client.get_converted_model(
                conversion_task_uuid=conversion_result.convert_task_uuid, access_token=self.token_handler.tokens.access_token
            )
            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download converted model failed. Error: {e}")
            raise e
