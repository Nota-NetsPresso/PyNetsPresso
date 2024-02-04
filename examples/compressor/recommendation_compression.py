from netspresso import NetsPresso
from netspresso.enums import CompressionMethod, RecommendationMethod

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare compressor
compressor = netspresso.compressor()

# 2. Set variables for compression
INPUT_MODEL_PATH = "./examples/sample_models/graphmodule.pt"
OUTPUT_DIR = "./outputs/compressed/graphmodule_recommend"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.5

# 3. Run recommendation compression
compression_result = compressor.recommendation_compression(
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    input_shapes=INPUT_SHAPES,
)
