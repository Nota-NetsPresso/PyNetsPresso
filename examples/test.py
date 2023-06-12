from loguru import logger

from netspresso.compressor import ModelCompressor, Task, Framework
from netspresso.compressor.client.utils.enum import CompressionMethod, RecommendationMethod


EMAIL = "bmlee@nota.ai"
PASSWORD = "qweasd"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

UPLOAD_MODEL_NAME = "test_h5"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.TENSORFLOW_KERAS
UPLOAD_MODEL_PATH = "./mobilenetv1.h5"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)
print(model)

models = compressor.get_models()
print(models)
uploaded_models = compressor.get_uploaded_models()
print(uploaded_models)
# compressed_models = compressor.get_compressed_models("db9e4228-64d2-4618-99fa-e8ae04842ea0")
# print(compressed_models)
# model = compressor.get_model("db9e4228-64d2-4618-99fa-e8ae04842ea0")
# print(model)

AUTO_COMPRESSED_MODEL_NAME = "test_auto_compress"
OUTPUT_PATH = "./mobilenetv1_cifar100_automatic.h5"
COMPRESSION_RATIO = 0.5
compressed_model = compressor.automatic_compression(
    model_id=model.model_id,
    model_name=AUTO_COMPRESSED_MODEL_NAME,
    output_path=OUTPUT_PATH,
    compression_ratio=COMPRESSION_RATIO,
)
print(compressed_model)

COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.5
OUTPUT_PATH = "./mobilenetv1_cifar100_recommend.h5"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
print(compressed_model)


COMPRESSION_METHOD = CompressionMethod.PR_L2
compression_1 = compressor.select_compression_method(
    model_id=model.model_id, compression_method=COMPRESSION_METHOD
)
logger.info(f"compression method: {compression_1.compression_method}")
logger.info(f"available layers: {compression_1.available_layers}")
for available_layer in compression_1.available_layers[:5]:
    available_layer.values = [0.2]
COMPRESSED_MODEL_NAME = "test_l2norm"
OUTPUT_PATH = "./mobilenetv1_cifar100_manual.h5"
compressed_model = compressor.compress_model(
    compression=compression_1,
    model_name=COMPRESSED_MODEL_NAME,
    output_path=OUTPUT_PATH,
)

# compression_info = compressor.get_compression("0113bc2f-b896-4599-8b7a-eb4ba46fa94c")
# print(compression_info)

UPLOAD_MODEL_NAME = "test_h5"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.TENSORFLOW_KERAS
UPLOAD_MODEL_PATH = "./mobilenetv1.h5"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)
AUTO_COMPRESSED_MODEL_NAME = "test_auto_compress"
OUTPUT_PATH = "./mobilenetv1_cifar100_automatic.h5"
COMPRESSION_RATIO = 0.5
compressed_model = compressor.automatic_compression(
    model_id=model.model_id,
    model_name=AUTO_COMPRESSED_MODEL_NAME,
    compression_ratio=COMPRESSION_RATIO,
    output_path=OUTPUT_PATH,
)
compressor.delete_model(model_id=model.model_id)
compressor.delete_model(model_id=model.model_id, recursive=True)
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)
compressor.delete_model(model_id=model.model_id)
