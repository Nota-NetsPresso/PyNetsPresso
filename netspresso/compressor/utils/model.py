from netspresso.compressor.core.model import CompressedModel, Model, ModelCollection


def get_model_object(model_info):
    model = Model(
        model_id=model_info.model_id,
        model_name=model_info.model_name,
        task=model_info.task,
        framework=model_info.framework,
        input_shapes=model_info.spec.input_layers,
        model_size=model_info.spec.model_size,
        flops=model_info.spec.flops,
        trainable_parameters=model_info.spec.trainable_parameters,
        non_trainable_parameters=model_info.spec.non_trainable_parameters,
        number_of_layers=model_info.spec.number_of_layers,
    )

    return model


def get_compressed_model_object(model_info):
    compressed_model = CompressedModel(
        model_id=model_info.model_id,
        model_name=model_info.model_name,
        task=model_info.task,
        framework=model_info.framework,
        input_shapes=model_info.spec.input_layers,
        model_size=model_info.spec.model_size,
        flops=model_info.spec.flops,
        trainable_parameters=model_info.spec.trainable_parameters,
        non_trainable_parameters=model_info.spec.non_trainable_parameters,
        number_of_layers=model_info.spec.number_of_layers,
        compression_id=model_info.original_compression_id,
        original_model_id=model_info.original_model_id,
    )

    return compressed_model


def get_model_collection_object(model_info):
    model_collection = ModelCollection(
        model_id=model_info.model_id,
        model_name=model_info.model_name,
        task=model_info.task,
        framework=model_info.framework,
        input_shapes=model_info.spec.input_layers,
        model_size=model_info.spec.model_size,
        flops=model_info.spec.flops,
        trainable_parameters=model_info.spec.trainable_parameters,
        non_trainable_parameters=model_info.spec.non_trainable_parameters,
        number_of_layers=model_info.spec.number_of_layers,
    )

    return model_collection
