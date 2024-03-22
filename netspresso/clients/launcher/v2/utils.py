from loguru import logger

from fastapi.security import OAuth2PasswordRequestForm

from netspresso.clients.auth.v2.client import auth_client_v2
from netspresso.clients.auth.v2.schemas.token import TokenResponse


def read_file_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        file_byte = f.read()
    return file_byte


def get_token_info(user_id: str, user_pw: str) -> TokenResponse:
    login_server_client = auth_client_v2
    response = login_server_client.login(
        request_body=OAuth2PasswordRequestForm(username=user_id, password=user_pw)
    )
    return response


def token_handler(inner_func):
    """
    :param
    :return:
    """

    def wrapper_func(*args, **kwargs):
        token_response = get_token_info(
            user_id=kwargs.pop("user_id"), user_pw=kwargs.pop("user_pw")
        )

        try:
            return inner_func(access_token=token_response.access_token, *args, **kwargs)
        except Exception as error:
            # TODO token 만료시 재발급 처리 추가
            logger.error(f"TokenHandler Error: {error}")

    return wrapper_func


def make_file_form_data_object(file_name, file_content):
    return [("file", (file_name, file_content))]
