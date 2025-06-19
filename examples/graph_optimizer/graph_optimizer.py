from loguru import logger

from netspresso.netspresso import NetsPresso

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

graph_optimizer = netspresso.graph_optimizer()

MODEL_PATH = "./examples/sample_models/yolo-fastest.onnx"
OUTPUT_DIR = "./outputs/graph_optimize/yolo-fastest"

optimized_model_metadata = graph_optimizer.optimize_model(input_model_path=MODEL_PATH, output_dir=OUTPUT_DIR)

logger.info(f"Start task result: {optimized_model_metadata.graph_optimize_task_info.graph_optimize_task_id}")
