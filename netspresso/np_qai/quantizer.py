import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import onnx
import qai_hub as hub
from loguru import logger
from qai_hub import JobStatus, QuantizeDtype
from qai_hub.client import Dataset, QuantizeJob
from qai_hub.public_rest_api import DatasetEntries

from netspresso.analytics import netspresso_analytics
from netspresso.enums import Status
from netspresso.metadata.quantizer import NPQAIQuantizerMetadata
from netspresso.np_qai.base import NPQAIBase
from netspresso.np_qai.options.quantize import QuantizeOptions
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


class NPQAIQuantizer(NPQAIBase):
    def get_quantize_task_status(self, quantize_task_id: str) -> JobStatus:
        """
        Get the status of a quantize task.

        Args:
            quantize_task_id: The ID of the quantize task to get the status of.

        Returns:
            JobStatus: The status of the quantize task.

        Note:
            For details, see [JobStatus in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.JobStatus.html).
        """
        job: QuantizeJob = hub.get_job(quantize_task_id)
        status = job.get_status()

        return status

    def update_quantize_task(self, metadata: NPQAIQuantizerMetadata) -> NPQAIQuantizerMetadata:
        """
        Update the quantize task.

        Args:
            metadata: The metadata of the quantize task.

        Returns:
            NPQAIQuantizerMetadata: The updated metadata of the quantize task.
        """
        job: QuantizeJob = hub.get_job(metadata.quantize_info.quantize_task_uuid)
        status = job.wait()

        if status.success:
            logger.info(f"{status.symbol} {status.state.name}")
            actual_model_path = self.download_model(job=job, filename=metadata.quantized_model_path)
            metadata.quantized_model_path = actual_model_path
            target_model = job.get_target_model()
            metadata.quantize_info.output_model_uuid = target_model.model_id
            metadata.status = Status.COMPLETED
        elif status.failure:
            logger.info(f"{status.symbol} {status.state}: {status.message}")
            metadata.status = Status.ERROR
            metadata.update_message(exception_detail=status.message)

        MetadataHandler.save_metadata(data=metadata, folder_path=Path(metadata.quantized_model_path).parent.as_posix())

        return metadata

    def quantize_model(
        self,
        input_model_path: Union[str, Path],
        output_dir: str,
        weights_dtype: QuantizeDtype,
        activations_dtype: QuantizeDtype,
        options: Union[QuantizeOptions, str] = QuantizeOptions(),
        job_name: Optional[str] = None,
        calibration_data: Union[Dataset, DatasetEntries, str, None] = None,
    ) -> Union[NPQAIQuantizerMetadata, List[NPQAIQuantizerMetadata]]:
        """
        Quantize a model in the QAI hub.

        Args:
            input_model_path: The path to the input model.
            output_dir: The directory to save the quantized model.
            weights_dtype: The data type to use for the weights.
            activations_dtype: The data type to use for the activations.
            options: The options to use for the quantization.
            job_name: The name of the job.
            calibration_data: The calibration data to use for the quantization.

        Returns:
            Union[NPQAIQuantizerMetadata, List[NPQAIQuantizerMetadata]]: Returns a quantizer metadata object if successful.

        Note:
            For details, see [submit_quantize_job in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_quantize_job.html).
        """
        netspresso_analytics.send_event(
            event_name="quantize_model_using_qai",
            event_params={
                "weights_dtype": weights_dtype.name if weights_dtype else "",
                "activations_dtype": activations_dtype.name if activations_dtype else "",
            },
        )

        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)
        default_model_path = (Path(output_dir) / f"{Path(output_dir).name}.ext").resolve()
        metadata = NPQAIQuantizerMetadata()
        metadata.input_model_path = Path(input_model_path).resolve().as_posix()
        extension = self.get_source_extension(model_path=input_model_path)
        metadata.model_info.framework = self.get_framework(extension=extension)

        MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        try:
            quantized_model_path = default_model_path.with_suffix(".onnx").as_posix()

            cli_string = options.to_cli_string() if isinstance(options, QuantizeOptions) else options

            job = hub.submit_quantize_job(
                model=input_model_path,
                calibration_data=calibration_data,
                weights_dtype=weights_dtype,
                activations_dtype=activations_dtype,
                name=job_name,
                options=cli_string,
            )
            metadata.quantized_model_path = quantized_model_path
            metadata.quantize_info.quantize_task_uuid = job.job_id
            metadata.quantize_info.input_model_uuid = job.model.model_id

            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        except KeyboardInterrupt:
            metadata.status = Status.STOPPED
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

    def download_model(self, job: QuantizeJob, filename: str) -> str:
        """
        Download a model from the QAI Hub and handle .onnx.zip conversion.

        Args:
            job: The job to download the model from.
            filename: The filename to save the model to.

        Returns:
            str: The actual path of the downloaded and processed model.

        Note:
            For details, see [download_target_model in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.QuantizeJob.html#qai_hub.QuantizeJob.download_target_model).
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
