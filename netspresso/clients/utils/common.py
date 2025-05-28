from pathlib import Path

from requests_toolbelt.multipart.encoder import MultipartEncoder
from tqdm import tqdm

from netspresso.clients.utils.system import ENV_STR

version = (Path(__file__).parent.parent.parent / "VERSION").read_text().strip()


def get_headers(access_token=None, json_type=False):
    headers = {"User-Agent": f"NetsPresso Python Package v{version} ({ENV_STR})"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if json_type:
        headers["Content-Type"] = "application/json"
    return headers


def read_file_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        file_byte = f.read()
    return file_byte


def get_files(file_path):
    return [
        (
            "file",
            (Path(file_path).name, open(file_path, "rb"), "application/octet-stream"),
        )
    ]


def create_multipart_data(url, file_info):
    # Prepare the multipart form data
    file_name = file_info[0]
    file_content = file_info[1]
    multipart_data = MultipartEncoder(
        fields={
            "url": (None, url, "application/json"),
            "file": (file_name, file_content, "application/octet-stream"),
        }
    )

    return multipart_data


def create_progress_func(multipart_data):
    # Progress callback function
    progress = tqdm(
        total=multipart_data.len,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        colour="#1BBFD6",
        desc="Uploading model",
    )

    return progress


def progress_callback(monitor, progress):
    progress.update(monitor.bytes_read - progress.n)
