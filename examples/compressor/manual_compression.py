from netspresso import NetsPresso
from netspresso.enums import CompressionMethod, GroupPolicy, LayerNorm, Options, Policy

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare compressor
compressor = netspresso.compressor()

# 2. Upload model
INPUT_MODEL_PATH = "./examples/sample_models/graphmodule.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]
model = compressor.upload_model(
    input_model_path=INPUT_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)

# 3. Select compression method
COMPRESSION_METHOD = CompressionMethod.PR_L2
OPTIONS = Options(
    policy=Policy.AVERAGE,
    layer_norm=LayerNorm.STANDARD_SCORE,
    group_policy=GroupPolicy.AVERAGE,
    reshape_channel_axis=-1,
)
compression_info = compressor.select_compression_method(
    model_id=model.model_id,
    compression_method=COMPRESSION_METHOD,
    options=OPTIONS,
)
print(f"compression method: {compression_info.compression_method}")
print(f"available layers: {compression_info.available_layers}")

# 4. Set params for compression(ratio or rank)
for available_layer in compression_info.available_layers[:5]:
    available_layer.values = [0.2]

# 5. Compress model
OUTPUT_DIR = "./outputs/compressed/graphmodule_manual"
compressed_model = compressor.compress_model(
    compression=compression_info,
    output_dir=OUTPUT_DIR,
)
