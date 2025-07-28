import json

import onnx
from google.protobuf.json_format import MessageToJson, Parse

from netspresso.exceptions.graph_optimizer import LoadONNXModelException, UpdateOnnxException


def update_dim_value(onnx_json, idx_list, key):
    for idx in idx_list:
        tensor_dim = onnx_json["graph"][key][idx]["type"]["tensorType"]["shape"]["dim"][0]
        tensor_dim["dimValue"] = 1
        if tensor_dim.get("dimParam"):
            del tensor_dim["dimParam"]
    return onnx_json


def find_initializer_idx_list_onnx(onnx_json):
    graph = onnx_json["graph"]
    not_filtered_inputs = graph["input"]  # it contains initialized variables

    initializers = graph["initializer"]
    initializer_names = [x["name"] for x in initializers]

    filtered_input_idx_list = []
    for i in range(len(not_filtered_inputs)):
        if graph["input"][i]["name"] not in initializer_names:
            filtered_input_idx_list.append(i)

    return filtered_input_idx_list


def load_onnx_model(model_path):
    try:
        onnx_model = onnx.load(model_path)
    except Exception as e:
        raise LoadONNXModelException(error_log=e.__repr__()) from e

    return onnx_model


def update_onnx_input_batch_size_as_1(model_path, save_path):
    onnx_model = load_onnx_model(model_path=model_path)

    try:
        s = MessageToJson(onnx_model)

        # set batch size as 1
        onnx_json = json.loads(s)

        # find input
        input_idx_list = find_initializer_idx_list_onnx(onnx_json)
        onnx_json = update_dim_value(onnx_json, input_idx_list, "input")
        output_idx_list = list(range(len(onnx_json["graph"]["output"])))
        onnx_json = update_dim_value(onnx_json, output_idx_list, "output")

        onnx_str = json.dumps(onnx_json)
        converted_model = Parse(onnx_str, onnx.ModelProto())
        onnx.save(converted_model, save_path)

        return save_path

    except Exception as e:
        raise UpdateOnnxException(error_log=e.__repr__()) from e
