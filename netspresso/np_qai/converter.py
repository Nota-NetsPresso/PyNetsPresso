import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import onnx
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

    def download_model(self, job: CompileJob, filename: str) -> str:
        """
        Download a model from the QAI Hub and handle .onnx.zip conversion.

        Args:
            job: The job to download the model from.
            filename: The filename to save the model to.

        Returns:
            str: The actual path of the downloaded and processed model.

        Note:
            For details, see [download_target_model in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.CompileJob.html#qai_hub.CompileJob.download_target_model).
            Since QAI Hub update, ONNX models are downloaded as .onnx.zip files.
            This function automatically extracts and converts them to .onnx format.
        """
        # Download the model (QAI Hub may save it as .zip file for ONNX models)
        downloaded_filename = job.download_target_model(filename=filename)

        # Use the original filename as fallback if download doesn't return a path
        if downloaded_filename is None:
            downloaded_filename = filename

        # Check if the downloaded file exists and is a zip file
        if os.path.exists(downloaded_filename) and zipfile.is_zipfile(downloaded_filename):
            logger.info(f"Downloaded file is a zip archive: {downloaded_filename}")
            logger.info("Extracting model from zip file...")

            # Extract the zip file
            extract_dir = f"{downloaded_filename}_extracted"
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(downloaded_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find the model file in the extracted directory (recursively search subdirectories)
            model_files = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith(('.onnx', '.tflite', '.so', '.bin')):
                        model_files.append(os.path.join(root, file))
                        break
                if model_files:
                    break

            if not model_files:
                raise FileNotFoundError(f"No model file found in extracted zip: {downloaded_filename}")

            # Load and save the model to the desired location
            extracted_model_path = model_files[0]

            # Determine the final path based on the model type
            if extracted_model_path.endswith('.onnx'):
                # For ONNX models, reload and save to ensure proper format
                onnx_model = onnx.load(extracted_model_path)

                # Save to the original filename with .onnx extension
                # Remove .zip extension first
                final_path = downloaded_filename.replace('.zip', '')
                # Remove duplicate .onnx if present (e.g., .onnx.onnx -> .onnx)
                if final_path.endswith('.onnx.onnx'):
                    final_path = final_path[:-5]  # Remove one .onnx to leave just one
                # Ensure it ends with .onnx
                elif not final_path.endswith('.onnx'):
                    final_path = f"{final_path}.onnx"

                onnx.save(onnx_model, final_path)
            else:
                # For non-ONNX models, just copy the file
                final_path = downloaded_filename.replace('.zip', '')
                shutil.copy2(extracted_model_path, final_path)

            logger.info(f"Model saved to: {final_path}")

            # Clean up temporary files
            os.remove(downloaded_filename)
            shutil.rmtree(extract_dir)

            return final_path

        # If not a zip file, return the downloaded filename as is
        return downloaded_filename
