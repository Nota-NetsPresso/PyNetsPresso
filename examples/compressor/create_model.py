from netspresso import NetsPresso
from netspresso.enums import Framework

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare compressor
compressor = netspresso.compressor()

# 2. Set variables for upload
FRAMEWORK = Framework.TENSORFLOW_KERAS
INPUT_MODEL_PATH = "./examples/sample_models/mobilenetv1.h5"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]

# 3. Upload model
model = compressor.upload_model(
    input_model_path=INPUT_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
    framework=FRAMEWORK,
)

# 4. Get Model
model = compressor.get_model(model_id=model.model_id)
print(model)
