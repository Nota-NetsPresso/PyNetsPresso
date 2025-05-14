from pathlib import Path
from typing import List, Optional, Union

import qai_hub as hub
from qai_hub.client import Dataset, Device, Job, JobStatus, JobSummary, JobType, Model, SourceModel, SourceModelType

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
            For details, see `upload_dataset in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.upload_dataset.html>`_.
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
            For details, see `get_dataset in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_dataset.html>`_.
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
            For details, see `get_datasets in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_datasets.html>`_.
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
            For details, see `upload_model in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.upload_model.html>`_.
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
            For details, see `get_models in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_models.html>`_.
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
            For details, see `get_model in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_model.html>`_.
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
            For details, see `get_devices in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_devices.html>`_.
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
            For details, see `get_device_attributes in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_device_attributes.html>`_.
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
            For details, see `get_job_summaries in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_job_summaries.html>`_.
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
            For details, see `get_job in QAI Hub API <https://app.aihub.qualcomm.com/docs/hub/generated/qai_hub.get_job.html>`_.
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
            Runtime.QNN_LIB_AARCH64_ANDROID: ".so",
            Runtime.QNN_CONTEXT_BINARY: ".bin",
            Runtime.ONNX: ".onnx",
            Runtime.PRECOMPILED_QNN_ONNX: ".zip",
        }

        return runtime_extensions.get(runtime)

    def get_display_runtime(self, runtime: Runtime) -> str:
        RUNTIME_DISPLAY_MAP = {
            Runtime.TFLITE: "TensorFlow Lite",
            Runtime.QNN_LIB_AARCH64_ANDROID: "Qualcomm® AI Engine Direct model library targeting AArch64 Android",
            Runtime.QNN_CONTEXT_BINARY: "Qualcomm® AI Engine Direct context binary targeting the hardware specified in the compile job.",
            Runtime.ONNX: "ONNX",
            Runtime.PRECOMPILED_QNN_ONNX: "ONNX Runtime model with a pre-compiled QNN context binary.",
        }
        return RUNTIME_DISPLAY_MAP.get(runtime, "Unknown runtime")

    def get_framework_by_runtime(self, runtime: Runtime):
        FRAMEWORK_RUNTIME_MAP = {
            Runtime.TFLITE: Framework.TFLITE,
            Runtime.QNN_LIB_AARCH64_ANDROID: Framework.QNN,
            Runtime.QNN_CONTEXT_BINARY: Framework.QNN,
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
