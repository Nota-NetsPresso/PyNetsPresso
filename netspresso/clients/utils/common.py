from pathlib import Path

from netspresso.clients.utils.system import ENV_STR

version = (Path(__file__).parent.parent.parent / "VERSION").read_text().strip()


def get_headers(access_token=None, json_type=False):
    headers = {"User-Agent": f"NetsPresso Python Package v{version} ({ENV_STR})"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if json_type:
        headers["Content-Type"] = "application/json"
    return headers


def create_tao_headers(token):
    return {"Authorization": f"Bearer {token}"}


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
