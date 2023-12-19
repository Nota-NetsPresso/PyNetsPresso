from netspresso.compressor import (
    ModelCompressor,
    Task,
    Framework,
    CompressionMethod,
    RecommendationMethod,
)


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

# Upload Model
MODEL_NAME = "test_pt"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.PYTORCH
INPUT_MODEL_PATH = "./examples/sample_models/graphmodule.pt"
OUTPUT_MODEL_PATH = "./outputs/compressed/graphmodule_recommend.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.5

compressed_model = compressor.recommendation_compression(
    model_name=MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    input_path=INPUT_MODEL_PATH,
    output_path=OUTPUT_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)
