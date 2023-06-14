from os.path import basename
from netspresso import __version__


def get_headers(access_token=None, json_type=False):
    headers = {
        "User-Agent": f"NetsPresso Model Compressor API Client v{__version__}",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if json_type:
        headers["Content-Type"] = "application/json"
    return headers


def get_files(file_path):
    return [("file", (basename(file_path), open(file_path, "rb"), "application/octet-stream"))]
