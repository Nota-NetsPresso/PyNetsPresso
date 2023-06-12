from loguru import logger

from netspresso.compressor import ModelCompressor


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

uploaded_models = compressor.get_uploaded_models()
logger.info(uploaded_models)

for uploaded_model in uploaded_models:
    compressed_models = compressor.get_compressed_models(uploaded_model["model_id"])
    logger.info(compressed_models)

models = compressor.get_models()
logger.info(models)
