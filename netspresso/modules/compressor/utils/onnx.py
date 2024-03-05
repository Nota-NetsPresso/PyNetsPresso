from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor


def _export_onnx(
    model: nn.Module,
    save_path: Union[str, Path],
    sample_input: Tensor,
    opset_version=13,
    input_names="images",
    output_names="output",
):
    torch.onnx.export(
        model,  # model being run
        sample_input,  # model input (or a tuple for multiple inputs)
        save_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[input_names],  # the model's input names
        output_names=[output_names],  # the model's output names
        dynamic_axes={input_names: {0: "batch_size"}, output_names: {0: "batch_size"}},  # variable length axes
    )
    logger.info(f"ONNX model converting and saved at {save_path}")


def export_onnx(file_path: str, input_shapes: List[int]):
    file_path = Path(file_path)
    model = torch.load(file_path.with_suffix(".pt"))

    input_shape = input_shapes[0]
    sample_input = torch.randn((1, input_shape.channel, *input_shape.dimension))
    save_dtype = next(model.parameters()).dtype

    return _export_onnx(model, file_path.with_suffix(".onnx"), sample_input=sample_input.type(save_dtype))
