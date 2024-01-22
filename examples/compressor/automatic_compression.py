from netspresso.compressor import Compressor


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = Compressor(email=EMAIL, password=PASSWORD)

MODEL_NAME = "test_graphmodule_pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]
INPUT_MODEL_PATH = "./examples/sample_models/graphmodule.pt"
OUTPUT_MODEL_PATH = "./outputs/compressed/graphmodule_automatic_compression"
COMPRESSION_RATIO = 0.5

compressed_model = compressor.automatic_compression(
    model_name=MODEL_NAME,
    input_shapes=INPUT_SHAPES,
    input_path=INPUT_MODEL_PATH,
    output_path=OUTPUT_MODEL_PATH,
    compression_ratio=COMPRESSION_RATIO,
)
