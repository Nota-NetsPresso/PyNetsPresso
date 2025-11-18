import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Union

import onnx
import qai_hub as hub
from loguru import logger
from qai_hub.client import (
    CompileJob,
    Dataset,
    Device,
    Job,
    JobStatus,
    JobSummary,
    JobType,
    Model,
    QuantizeJob,
    SourceModel,
    SourceModelType,
)

from netspresso.np_qai.options import Extension, Framework, Runtime


class NPQAIBase:
    def set_verbose(self, verbose: bool) -> None:
        hub.set_verbose(verbose)

    def upload_dataset(self, data, name=None) -> Dataset:
        """
        Upload a dataset to the QAI Hub.

        Args:
            data: The dataset to upload.
            name: The name of the dataset.

        Returns:
            Dataset: Returns a dataset object if successful.

        Note:
            For details, see [upload_dataset in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.upload_dataset.html).
        """
        dataset = hub.upload_dataset(data=data, name=name)

        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        """
        Get a dataset from the QAI Hub.

        Args:
            dataset_id: The ID of the dataset to get.

        Returns:
            Dataset: Returns a dataset object if successful.

        Note:
            For details, see [get_dataset in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_dataset.html).
        """
        dataset = hub.get_dataset(dataset_id=dataset_id)

        return dataset

    def get_datasets(self, offset: int = 0, limit: int = 50) -> List[Dataset]:
        """
        Get a list of datasets from the QAI Hub.

        Args:
            offset: The offset of the datasets to get even older datasets.
            limit: The limit of the datasets to get.

        Returns:
            List[Dataset]: Returns a list of dataset objects if successful.

        Note:
            For details, see [get_datasets in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_datasets.html).
        """
        datasets = hub.get_datasets(offset=offset, limit=limit)

        return datasets

    def upload_model(self, model: Union[SourceModel, str], name: Optional[str] = None) -> Model:
        """
        Upload a model to the QAI Hub.

        Args:
            model: The model to upload.
            name: The name of the model.

        Returns:
            Model: Returns a model object if successful.

        Note:
            For details, see [upload_model in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.upload_model.html).
        """
        model = hub.upload_model(model=model, name=name)

        return model

    def get_models(self, offset: int = 0, limit: int = 50) -> List[Model]:
        """
        Get a list of models from the QAI Hub.

        Args:
            offset: The offset of the models to get even older models.
            limit: The limit of the models to get.

        Returns:
            List[Model]: Returns a list of model objects if successful.

        Note:
            For details, see [get_models in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_models.html).
        """
        models = hub.get_models(offset=offset, limit=limit)

        return models

    def get_model(self, model_id: str) -> Model:
        """
        Get a model from the QAI Hub.

        Args:
            model_id: The ID of the model to get.

        Returns:
            Model: Returns a model object if successful.

        Note:
            For details, see [get_model in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_model.html).
        """
        model = hub.get_model(model_id=model_id)

        return model

    def get_devices(self, name: str = "", os: str = "", attributes: Union[str, List[str]] = None) -> List[Device]:
        """
        Get a list of devices from the QAI Hub.

        Args:
            name: The name of the device to get.
            os: The OS of the device to get.
            attributes: The attributes of the device to get.

        Returns:
            List[Device]: Returns a list of device objects if successful.

        Note:
            For details, see [get_devices in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_devices.html).
        """
        if attributes is None:
            attributes = []
        devices = hub.get_devices(name=name, os=os, attributes=attributes)

        return devices

    def get_device_attributes(self) -> List[str]:
        """
        Get a list of device attributes from the QAI Hub.

        Returns:
            List[str]: Returns a list of device attribute strings if successful.

        Note:
            For details, see [get_device_attributes in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_device_attributes.html).
        """
        device_attributes = hub.get_device_attributes()

        return device_attributes

    def get_job_summaries(
        self,
        offset: int = 0,
        limit: int = 50,
        creator: Optional[str] = None,
        state: Union[Optional[JobStatus.State], List[JobStatus.State]] = None,
        type: Optional[JobType] = None,
    ) -> List[JobSummary]:
        """
        Get a list of job summaries from the QAI Hub.

        Returns:
            List[JobSummary]: Returns a list of job summary objects if successful.

        Note:
            For details, see [get_job_summaries in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_job_summaries.html).
        """
        job_summaries = hub.get_job_summaries(offset=offset, limit=limit, creator=creator, state=state, type=type)

        return job_summaries

    def get_job(self, job_id: str) -> Job:
        """
        Get a job from the QAI Hub.

        Args:
            job_id: The ID of the job to get.

        Returns:
            Job: Returns a job object if successful.

        Note:
            For details, see [get_job in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_job.html).
        """
        job = hub.get_job(job_id=job_id)

        return job

    def get_source_extension(self, model_path):
        extension = Path(model_path).suffix

        return extension

    def get_framework(self, extension: Extension):
        if extension == Extension.ONNX:
            return Framework.ONNX
        elif extension == Extension.PT:
            return Framework.PYTORCH
        elif extension == Extension.AIMET:
            return Framework.AIMET
        elif extension == Extension.H5:
            return Framework.TENSORFLOW

    def get_target_extension(self, runtime=Runtime.TFLITE):
        runtime_extensions = {
            Runtime.TFLITE: ".tflite",
            Runtime.QNN_LIB_AARCH64_ANDROID: ".so",  # Deprecated in qai-hub 0.40.0
            Runtime.QNN_CONTEXT_BINARY: ".bin",
            Runtime.QNN_DLC: ".dlc",  # Added in qai-hub 0.40.0
            Runtime.ONNX: ".onnx",
            Runtime.PRECOMPILED_QNN_ONNX: ".zip",
        }

        return runtime_extensions.get(runtime)

    def get_display_runtime(self, runtime: Runtime) -> str:
        RUNTIME_DISPLAY_MAP = {
            Runtime.TFLITE: "TensorFlow Lite",
            Runtime.QNN_LIB_AARCH64_ANDROID: "Qualcomm® AI Engine Direct model library targeting AArch64 Android (Deprecated - use QNN_DLC)",
            Runtime.QNN_CONTEXT_BINARY: "Qualcomm® AI Engine Direct context binary targeting the hardware specified in the compile job.",
            Runtime.QNN_DLC: "Qualcomm® AI Engine Direct DLC (Deep Learning Container) - Recommended for QNN deployment",
            Runtime.ONNX: "ONNX",
            Runtime.PRECOMPILED_QNN_ONNX: "ONNX Runtime model with a pre-compiled QNN context binary.",
        }
        return RUNTIME_DISPLAY_MAP.get(runtime, "Unknown runtime")

    def get_framework_by_runtime(self, runtime: Runtime):
        FRAMEWORK_RUNTIME_MAP = {
            Runtime.TFLITE: Framework.TFLITE,
            Runtime.QNN_LIB_AARCH64_ANDROID: Framework.QNN,
            Runtime.QNN_CONTEXT_BINARY: Framework.QNN,
            Runtime.QNN_DLC: Framework.QNN,  # Added in qai-hub 0.40.0
            Runtime.ONNX: Framework.ONNX,
            Runtime.PRECOMPILED_QNN_ONNX: Framework.QNN,
        }
        return FRAMEWORK_RUNTIME_MAP.get(runtime, "Unknown framework")

    def get_framework_by_model_type(self, model_type: SourceModelType):
        FRAMEWORK_MODEL_TYPE_MAP = {
            SourceModelType.TORCHSCRIPT: Framework.PYTORCH,
            SourceModelType.TFLITE: Framework.TFLITE,
            SourceModelType.ONNX: Framework.ONNX,
            SourceModelType.MLMODEL: Framework.COREML,
            SourceModelType.MLMODELC: Framework.COREML,
            SourceModelType.MLPACKAGE: Framework.COREML,
            SourceModelType.TETRART: Framework.TENSORRT,
            SourceModelType.QNN_LIB_AARCH64_ANDROID: Framework.QNN,
            SourceModelType.QNN_LIB_X86_64_LINUX: Framework.QNN,
            SourceModelType.QNN_CONTEXT_BINARY: Framework.QNN,
            SourceModelType.AIMET_ONNX: Framework.AIMET,
        }
        return FRAMEWORK_MODEL_TYPE_MAP.get(model_type, "Unknown framework")

    def get_display_framework(self, framework: Framework):
        RUNTIME_DISPLAY_MAP = {
            Framework.PYTORCH: "PyTorch",
            Framework.ONNX: "ONNX",
            Framework.ONNXRUNTIME: "ONNXRuntime",
            Framework.AIMET: "AIMET",
            Framework.TENSORFLOW: "TensorFlow",
            Framework.TFLITE: "Tensorflow Lite",
            Framework.COREML: "CoreML",
            Framework.TENSORRT: "TensorRT",
            Framework.QNN: "QNN",
        }
        return RUNTIME_DISPLAY_MAP.get(framework, "Unknown runtime")

    def download_model(self, job: Union[CompileJob, QuantizeJob], filename: str) -> str:
        """
        Download a model from the QAI Hub and handle ONNX zip conversion.

        Args:
            job: The job to download the model from (CompileJob or QuantizeJob).
            filename: The filename to save the model to.

        Returns:
            str: The actual path of the downloaded and processed model.

        Note:
            Since QAI Hub July 2025 update, ONNX models are always downloaded as .zip files
            with external weights. Other formats (.tflite, .so, .bin) are downloaded directly.
            Reference: https://app.aihub.qualcomm.com/docs/hub/release_notes.html#released-july-14-2025

            This function automatically extracts and converts ONNX zip files to single .onnx files
            with embedded weights. Non-ONNX models are returned as-is.
        """
        # Download the model
        downloaded_filename = job.download_target_model(filename=filename)

        # Use the original filename as fallback if download doesn't return a path
        if downloaded_filename is None:
            downloaded_filename = filename

        # Verify file exists
        if not os.path.exists(downloaded_filename):
            raise FileNotFoundError(f"Downloaded file not found: {downloaded_filename}")

        # Only ONNX models are downloaded as .zip (since QAI Hub July 2025 update)
        # Other formats (.tflite, .so, .bin) are downloaded directly
        is_onnx_zip = downloaded_filename.endswith('.onnx.zip') or (
            zipfile.is_zipfile(downloaded_filename) and '.onnx' in filename.lower()
        )

        if is_onnx_zip:
            return self._extract_and_convert_onnx_zip(downloaded_filename)

        # Non-ONNX models: return as-is
        return downloaded_filename

    def _extract_and_convert_onnx_zip(self, zip_path: str) -> str:
        """
        Extract and convert ONNX model from zip file to single .onnx file with embedded weights.

        Args:
            zip_path: Path to the .onnx.zip file.

        Returns:
            str: Path to the converted .onnx file.

        Note:
            Since QAI Hub July 2025 update, ONNX models are always produced with external
            weights in .zip format. This method extracts the zip and saves as a single .onnx file.
            Reference: https://app.aihub.qualcomm.com/docs/hub/release_notes.html#released-july-14-2025
        """
        logger.info(f"Extracting ONNX model from zip: {zip_path}")

        extract_dir = None
        try:
            # Create temporary extraction directory
            extract_dir = f"{zip_path}_extracted"
            os.makedirs(extract_dir, exist_ok=True)

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find ONNX file (recursively search subdirectories)
            onnx_files = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.onnx'):
                        onnx_files.append(os.path.join(root, file))

            if not onnx_files:
                raise FileNotFoundError(f"No ONNX file found in zip: {zip_path}")

            if len(onnx_files) > 1:
                logger.warning(f"Multiple ONNX files found in zip, using first: {onnx_files[0]}")

            # Load ONNX model
            extracted_onnx_path = onnx_files[0]
            onnx_model = onnx.load(extracted_onnx_path)

            # Determine final path: remove .zip extension
            if zip_path.endswith('.onnx.zip'):
                final_path = zip_path[:-4]  # Remove .zip, keep .onnx
            elif zip_path.endswith('.zip'):
                final_path = zip_path[:-4] + '.onnx'
            else:
                final_path = f"{zip_path}.onnx"

            # Save as single ONNX file with embedded weights
            onnx.save(onnx_model, final_path)
            logger.info(f"ONNX model saved to: {final_path}")

            return final_path

        finally:
            # Always cleanup temporary files
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if extract_dir and os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary files: {e}")
