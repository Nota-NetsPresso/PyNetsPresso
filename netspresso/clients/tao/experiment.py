from netspresso.clients.utils.requester import Requester


class ExperimentAPI:
    def __init__(self, url: str):
        self.url = url

    def get_experiments(self, user_id, skip, size, sort, name, type, arch, read_only, user_only, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments"
        params = {
            "skip": skip,
            "size": size,
            "sort": sort,
            "name": name,
            "type": type,
            "arch": arch,
            "read_only": read_only,
            "user_only": user_only,
        }

        response = Requester.get(url=endpoint, params=params, headers=headers)

        return response.json()

    def create_experiments(self, user_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments"

        response = Requester.post_as_json(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def delete_experiment(self, user_id, experiment_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}"

        response = Requester.delete(url=endpoint, headers=headers)

        return response.json()

    def get_experiment(self, user_id, experiment_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def partial_update_experiment(self, user_id, experiment_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}"

        response = Requester.patch(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def update_experiment(self, user_id, experiment_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}"

        response = Requester.put(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def get_experiment_jobs(self, user_id, experiment_id, skip, size, sort, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs"
        params = {"skip": skip, "size": size, "sort": sort}

        response = Requester.get(url=endpoint, params=params, headers=headers)

        return response.json()

    def run_experiment_jobs(self, user_id, experiment_id, request_body, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs"

        response = Requester.post_as_json(url=endpoint, request_body=request_body, headers=headers)

        return response.json()

    def delete_experiment_job(self, user_id, experiment_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs/{job_id}"

        response = Requester.delete(url=endpoint, headers=headers)

        return response.json()

    def get_experiment_job(self, user_id, experiment_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs/{job_id}"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def cancel_experiment_job(self, user_id, experiment_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs/{job_id}:cancel"

        response = Requester.post_as_json(url=endpoint, headers=headers)

        return response.json()

    def download_job_artifacts(self, user_id, experiment_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs/{job_id}:download"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()

    def resume_experiment_job(self, user_id, experiment_id, job_id, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/jobs/{job_id}:resume"

        response = Requester.post_as_json(url=endpoint, headers=headers)

        return response.json()

    def get_specs_schema(self, user_id, experiment_id, action, headers):
        endpoint = f"{self.url}/users/{user_id}/experiments/{experiment_id}/specs/{action}/schema"

        response = Requester.get(url=endpoint, headers=headers)

        return response.json()