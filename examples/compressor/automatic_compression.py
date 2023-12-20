from netspresso.compressor import ModelCompressor, Task, Framework


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

MODEL_NAME = "test_h5"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.TENSORFLOW_KERAS
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
INPUT_MODEL_PATH = "./examples/sample_models/mobilenetv1.h5"
OUTPUT_MODEL_PATH = "./outputs/compressed/mobilenetv1_cifar100_automatic.h5"
COMPRESSION_RATIO = 0.5

compressed_model = compressor.automatic_compression(
    model_name=MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    input_shapes=INPUT_SHAPES,
    input_path=INPUT_MODEL_PATH,
    output_path=OUTPUT_MODEL_PATH,
    compression_ratio=COMPRESSION_RATIO,
)
