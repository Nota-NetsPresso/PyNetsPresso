import time
from pathlib import Path
from typing import Dict, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import BaseClient, validate_token
from netspresso.clients.launcher import LauncherAPIClient
from netspresso.enums import (
    DataType,
    DeviceName,
    Framework,
    Module,
    SoftwareVersion,
    TaskStatus,
)
from netspresso.clients.launcher.schemas.model import (
    ConversionTask,
    InputShape,
    Model,
    TargetDevice,
)
from netspresso.enums import ServiceCredit, TaskType, Status
from netspresso.clients.launcher.schemas import TargetDeviceFilter

from ..utils import FileHandler, check_credit_balance
from ..utils.metadata import MetadataHandler


class Converter(BaseClient):
    def __init__(self, email=None, password=None, user_session=None):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
            user_session (SessionClient): The SessionClient object.

        Available constructors:
            Converter(email='USER_EMAIL',password='PASSWORD')
            Converter(user_session=SessionClient(email='USER_EMAIL',password='PASSWORD')
        """
        super().__init__(email=email, password=password, user_session=user_session)
        self.client = LauncherAPIClient(user_sessoin=self.user_session)

    @validate_token
    def convert_model(
        self,
        model_path: Union[Path, str],
        output_path: Union[Path, str],
        target_framework: Union[str, Framework],
        data_type: DataType = DataType.FP16,
        target_device: TargetDevice = None,
        wait_until_done: bool = True,
        target_device_name: DeviceName = None,
        target_software_version: SoftwareVersion = None,
        input_shape: InputShape = None,
        dataset_path: str = None,
    ) -> Dict:
        """Convert a model into the type the specific framework.

        Args:
            model_path (str): The file path where the model is located.
            output_path (str): The local path to save the converted model.
            target_framework (Framework | str): the target framework name.
            data_type (DataType): data type of the model.
            target_device (TargetDevice): target device. If it's not set, target_device_name and target_software_version have to be set.
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.
            target_device_name (DeviceName): target device name. Necessary field if target_device is not given.
            target_software_version (SoftwareVersion): target_software_version. Necessary field if target_device_name is one of jetson devices.
            input_shape (InputShape) : target input shape to convert. (ex: dynamic batch to static batch)

        Raises:
            e: If an error occurs while converting the model.

        Returns:
            Dict: model conversion task dict.
        """
        try:
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_path, framework=target_framework
            )
            FileHandler.create_folder(folder_path=output_path)
            metadata = MetadataHandler.init_metadata(folder_path=output_path, task_type=TaskType.CONVERT)

            current_credit = self.user_session.get_credit()
            check_credit_balance(
                user_credit=current_credit, service_credit=ServiceCredit.MODEL_CONVERT
            )
            model = self.client.upload_model(model_file_path=model_path, target_function=Module.CONVERT)

            model_uuid = model
            if type(model) is Model:
                model_uuid = model.model_uuid
                if input_shape is None and model.input_shape is not None:
                    input_shape = model.input_shape
                if target_framework is None and model.framework is not None:
                    target_framework = model.framework

            if target_device is None:
                if target_device_name is None:
                    raise NotImplementedError(
                        "The conversion is unavailable. Please set target_device or target_device_name."
                    )

                elif (
                    target_device_name in DeviceName.JETSON_DEVICES and target_software_version is None
                ):
                    raise NotImplementedError(
                        "The conversion is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                    )

                if type(model) is not Model:
                    raise NotImplementedError(
                        "The conversion is unavailable. Please set target_device while using model's uuid string for the conversion."
                    )

                # Check available int8 converting devices
                if data_type == DataType.INT8:
                    if target_device_name not in DeviceName.AVAILABLE_INT8_DEVICES:
                        raise Exception(
                            f"int8 converting supports only {DeviceName.AVAILABLE_INT8_DEVICES}."
                        )
                else:  # FP16, FP32
                    if target_device_name in DeviceName.ONLY_INT8_DEVICES:
                        raise Exception(
                            f"{DeviceName.ONLY_INT8_DEVICES} only support int8 data types."
                        )

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

            logger.info(
                f"Converting Model for {target_device.device_name} ({target_framework})"
            )

            conversion_task = self.client.convert_model(
                model_uuid=model_uuid,
                input_shape=input_shape,
                target_framework=target_framework,
                target_device=target_device.device_name,
                data_type=data_type,
                software_version=target_device.software_version,
                dataset_path=dataset_path,
            )

            conversion_task = self.get_conversion_task(conversion_task)

            if wait_until_done:
                while conversion_task.status in [
                    TaskStatus.IN_QUEUE,
                    TaskStatus.IN_PROGRESS,
                ]:
                    conversion_task = self.get_conversion_task(conversion_task)
                    time.sleep(1)

            self.download_converted_model(
                conversion_task, default_model_path.with_suffix(extension)
            )

            converter_uploaded_model = self.client.upload_model(
                model_file_path=default_model_path.with_suffix(extension),
                target_function=Module.BENCHMARK,
            )

            metadata.update_converted_model_path(converted_model_path=default_model_path.with_suffix(extension).as_posix())
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
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_path)

            remaining_credit = self.user_session.get_credit()
            logger.info(
                f"{ServiceCredit.MODEL_CONVERT} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            return metadata.asdict()
        
        except Exception as e:
            logger.error(f"Convert failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_path)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_path)

    @validate_token
    def get_conversion_task(
        self, conversion_task: Union[str, ConversionTask]
    ) -> ConversionTask:
        """Get the conversion task information with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
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
            return self.client.get_conversion_task(
                conversion_task_uuid=conversion_task_uuid
            )

        except Exception as e:
            logger.error(f"Get conversion task failed. Error: {e}")
            raise e

    @validate_token
    def download_converted_model(
        self, conversion_task: Union[str, ConversionTask], local_path: str
    ):
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
        try:
            conversion_task_uuid = None
            if type(conversion_task) is str:
                conversion_task_uuid = conversion_task
            elif type(conversion_task) is ConversionTask:
                conversion_task_uuid = conversion_task.convert_task_uuid

            conversion_result: ConversionTask = self.get_conversion_task(
                conversion_task_uuid
            )
            if conversion_result.status is TaskStatus.ERROR:
                raise FileNotFoundError(
                    "The conversion is Failed. There is no file available for download."
                )
            if conversion_result.status is not TaskStatus.FINISHED:
                raise FileNotFoundError(
                    "The conversion is in progress. There is no file available for download at the moment."
                )

            download_url = self.client.get_converted_model(
                conversion_task_uuid=conversion_result.convert_task_uuid
            )
            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download converted model failed. Error: {e}")
            raise e
