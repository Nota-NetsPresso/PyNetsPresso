from os.path import basename


def get_headers(access_token=None, json_type=False):
    headers = {
        "User-Agent": "NetsPresso Model Compressor API Client",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if json_type:
        headers["Content-Type"] = "application/json"
    return headers


def get_files(file_path):
    return [("file", (basename(file_path), open(file_path, "rb"), "application/octet-stream"))]
