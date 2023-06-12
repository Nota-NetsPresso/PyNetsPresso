from netspresso.compressor import ModelCompressor, Task, Framework


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

UPLOAD_MODEL_NAME = "test_h5"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.TENSORFLOW_KERAS
UPLOAD_MODEL_PATH = "./mobilenetv1.h5"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]

# Upload Model
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)

# Compress Model
AUTO_COMPRESSED_MODEL_NAME = "test_auto_compress"
OUTPUT_PATH = "./mobilenetv1_cifar100_automatic.h5"
COMPRESSION_RATIO = 0.5
compressed_model = compressor.automatic_compression(
    model_id=model.model_id,
    model_name=AUTO_COMPRESSED_MODEL_NAME,
    output_path=OUTPUT_PATH,
    compression_ratio=COMPRESSION_RATIO,
)
