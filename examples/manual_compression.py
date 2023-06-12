from loguru import logger

from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

# Upload Model
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

# Select Compression Method
COMPRESSION_METHOD = CompressionMethod.PR_L2
compression_1 = compressor.select_compression_method(model_id=model.model_id, compression_method=COMPRESSION_METHOD)
logger.info(f"compression method: {compression_1.compression_method}")
logger.info(f"available layers: {compression_1.available_layers}")

# Set Compression Params
for available_layer in compression_1.available_layers[:5]:
    available_layer.values = [0.2]

# Compress Model
COMPRESSED_MODEL_NAME = "test_l2norm"
OUTPUT_PATH = "./mobilenetv1_cifar100_manual.h5"
compressed_model = compressor.compress_model(
    compression=compression_1,
    model_name=COMPRESSED_MODEL_NAME,
    output_path=OUTPUT_PATH,
)
