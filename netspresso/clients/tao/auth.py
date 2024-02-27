from netspresso.clients.utils.requester import Requester


class AuthAPI:
    def __init__(self, url: str):
        self.url = url

    def login(self, request_body):
        endpoint = f"{self.url}/login"

        response = Requester.post_as_json(url=endpoint, request_body=request_body)

        return response.json()
