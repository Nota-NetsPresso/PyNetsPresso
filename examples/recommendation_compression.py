from netspresso.compressor import (
    ModelCompressor,
    Task,
    Framework,
    CompressionMethod,
    RecommendationMethod,
    Policy,
    LayerNorm,
    GroupPolicy,
    Options,
)


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

# Upload Model
UPLOAD_MODEL_NAME = "test_pt"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./examples/sample_models/graphmodule.pt"
INPUT_SHAPES = [{"batch": 2, "channel": 3, "dimension": [224, 224]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)

# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.5
OUTPUT_PATH = "./graphmodule_recommend.pt"
OPTIONS = Options(
    policy=Policy.AVERAGE, layer_norm=LayerNorm.TSS_NORM, group_policy=GroupPolicy.COUNT, reshape_channel_axis=-1
)
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    options=OPTIONS,
    output_path=OUTPUT_PATH,
)
