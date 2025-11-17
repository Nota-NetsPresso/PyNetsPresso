from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from loguru import logger
from qai_hub import JobStatus
from qai_hub.client import CompileJob, Dataset, Device, InputSpecs
from qai_hub.public_rest_api import DatasetEntries

from netspresso.analytics import netspresso_analytics
from netspresso.enums import Status
from netspresso.metadata.converter import ConverterMetadata
from netspresso.np_qai.base import NPQAIBase
from netspresso.np_qai.options import CompileOptions
from netspresso.np_qai.options.common import normalize_device_name
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class NPQAIConverter(NPQAIBase):
    def convert_image_dict_to_list(self, image_dict):
        result = []
        for key, value in image_dict.items():
            batch, channel, *dimension = value
            result.append({"name": key, "batch": batch, "channel": channel, "dimension": dimension})
        return result

    def get_convert_task_status(self, convert_task_id: str) -> JobStatus:
        """
        Get the status of a convert task.

        Args:
            convert_task_id: The ID of the convert task to get the status of.

        Returns:
            JobStatus: The status of the convert task.

        Note:
            For details, see [JobStatus in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.JobStatus.html).
        """
        job: CompileJob = hub.get_job(convert_task_id)
        status = job.get_status()

        return status

    def update_convert_task(self, metadata: ConverterMetadata) -> ConverterMetadata:
        """
        Update the convert task.

        Args:
            metadata: The metadata of the convert task.

        Returns:
            ConverterMetadata: The updated metadata of the convert task.
        """
        job: CompileJob = hub.get_job(metadata.convert_task_info.convert_task_uuid)
        status = job.wait()

        if status.success:
            logger.info(f"{status.symbol} {status.state.name}")
            actual_model_path = self.download_model(job=job, filename=metadata.converted_model_path)
            metadata.converted_model_path = actual_model_path
            target_model = job.get_target_model()
            metadata.convert_task_info.output_model_uuid = target_model.model_id
            metadata.convert_task_info.data_type = job.target_shapes["image"][1]
            metadata.available_options = job.compatible_devices
            metadata.status = Status.COMPLETED
        elif status.failure:
            logger.info(f"{status.symbol} {status.state}: {status.message}")
            metadata.status = Status.ERROR
            metadata.update_message(exception_detail=status.message)

        MetadataHandler.save_metadata(data=metadata, folder_path=Path(metadata.converted_model_path).parent.as_posix())

        return metadata

    def convert_model(
        self,
        input_model_path: Union[str, Path],
        output_dir: str,
        target_device_name: Union[Device, List[Device]],
        input_shapes: Optional[InputSpecs] = None,
        options: Union[CompileOptions, str] = CompileOptions(),
        job_name: Optional[str] = None,
        single_compile: bool = True,
        calibration_data: Union[Dataset, DatasetEntries, str, None] = None,
        retry: bool = True,
    ) -> Union[ConverterMetadata, List[ConverterMetadata]]:
        """
        Convert a model in the QAI hub.

        Args:
            input_model_path: The path to the input model.
            output_dir: The directory to save the converted model.
            target_device_name: The device to compile the model for.
            input_shapes: The input shapes of the model.
            options: The options to use for the conversion.
            job_name: The name of the job.
            single_compile: Whether to compile the model in a single step.
            calibration_data: The calibration data to use for the conversion.
            retry: Whether to retry the conversion if it fails.

        Returns:
            Union[ConverterMetadata, List[ConverterMetadata]]: Returns a converter metadata object if successful.

        Note:
            For details, see [submit_compile_job in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html).
        """
        netspresso_analytics.send_event(
            event_name="convert_model_using_qai",
            event_params={
                "target_device_name": normalize_device_name(target_device_name) or "",
                "target_runtime": options.target_runtime or "",
                "quantize_full_type": options.quantize_full_type or "",
                "quantize_weight_type": options.quantize_weight_type or "",
                "compute_unit": options.normalize_compute_units() or "",
            },
        )

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        default_model_path = (Path(output_dir) / f"{Path(output_dir).name}.ext").resolve()
        metadata = ConverterMetadata()
        metadata.input_model_path = Path(input_model_path).resolve().as_posix()
        extension = self.get_source_extension(model_path=input_model_path)
        metadata.model_info.framework = self.get_framework(extension=extension)

        MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        try:
            target_extension = self.get_target_extension(runtime=options.target_runtime)
            converted_model_path = default_model_path.with_suffix(target_extension).as_posix()

            cli_string = options.to_cli_string() if isinstance(options, CompileOptions) else options

            job = hub.submit_compile_job(
                model=input_model_path,
                device=target_device_name,
                name=job_name,
                input_specs=input_shapes,
                options=cli_string,
                single_compile=single_compile,
                calibration_data=calibration_data,
                retry=retry,
            )

            framework = self.get_framework_by_runtime(options.target_runtime)
            display_framework = self.get_display_framework(framework)

            metadata.model_info.input_shapes = self.convert_image_dict_to_list(input_shapes)
            metadata.model_info.data_type = job.shapes["image"][1]
            metadata.convert_task_info.convert_task_uuid = job.job_id
            metadata.converted_model_path = converted_model_path
            metadata.convert_task_info.input_model_uuid = job.model.model_id
            metadata.convert_task_info.device_name = target_device_name.name
            metadata.convert_task_info.display_device_name = target_device_name.name
            metadata.convert_task_info.framework = framework
            metadata.convert_task_info.display_framework = display_framework

            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata
