from pathlib import Path

from netspresso.clients.utils.common import read_file_bytes
from netspresso.clients.utils.requester import Requester


class DatasetAPI:
    def __init__(self, url: str):
        self.url = url

    def get_datasets(self, user_id, headers, skip=None, size=None, sort=None, name=None, format=None, type=None):
        endpoint = f"{self.url}/users/{user_id}/datasets"
        params = {
            key: value
            for key, value in {
                "skip": skip,
                "size": size,
                "sort": sort,
                "name": name,
                "format": format,
                "type": type,
            }.items()
            if value is not None
        }

        response = Requester.get(url=endpoint, params=params, headers=headers)

        return response.json()

    def create_dataset(self, user_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets"

        response = Requester.post_as_json(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def delete_dataset(self, user_id, dataset_id, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}"

        response = Requester.delete(url=endpoint, headers=headers)

        return response.json()

    def get_dataset(self, user_id, dataset_id, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def partial_update_dataset(self, user_id, dataset_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}"

        response = Requester.patch(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def update_dataset(self, user_id, dataset_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}"

        response = Requester.put(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def get_dataset_jobs(self, user_id, dataset_id, headers, skip=None, size=None, sort=None):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/jobs"
        params = {
            key: value
            for key, value in {
                "skip": skip,
                "size": size,
                "sort": sort,
            }.items()
            if value is not None
        }

        response = Requester.get(url=endpoint, params=params, headers=headers)

        return response.json()

    def run_dataset_jobs(self, user_id, dataset_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/jobs"

        response = Requester.post_as_json(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def delete_dataset_job(self, user_id, dataset_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/jobs/{job_id}"

        response = Requester.delete(url=endpoint, headers=headers)

        return response.json()

    def get_dataset_job(self, user_id, dataset_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/jobs/{job_id}"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def cancel_dataset_job(self, user_id, dataset_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/jobs/{job_id}:cancel"

        response = Requester.post_as_json(url=endpoint, headers=headers)

        return response.json()

    def download_job_artifacts(self, user_id, dataset_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/jobs/{job_id}:download"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def get_specs_schema(self, user_id, dataset_id, action, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}/specs/{action}/schema"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def upload_dataset(self, user_id, dataset_id, dataset_path, headers):
        endpoint = f"{self.url}/users/{user_id}/datasets/{dataset_id}:upload"
        file_content = read_file_bytes(file_path=dataset_path)
        file_obj = [("file", (Path(dataset_path).name, file_content))]

        response = Requester.post_as_form(url=endpoint, binary=file_obj, headers=headers)

        return response.json()
