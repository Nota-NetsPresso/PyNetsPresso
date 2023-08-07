from loguru import logger

from netspresso.compressor import ModelCompressor, Task, Framework


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

UPLOAD_MODEL_NAME = "test_h5"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.TENSORFLOW_KERAS
UPLOAD_MODEL_PATH = "./examples/sample_models/mobilenetv1.h5"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]

# Upload Model
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)

# Get Model
model = compressor.get_model(model_id=model.model_id)
logger.info(f"model id: {model.model_id}")
