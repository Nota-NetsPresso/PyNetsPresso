__version__ = "1.0.5"

from functools import wraps
from netspresso.schemas.auth import LoginRequest, RefreshTokenRequest
from netspresso.utils.token import check_jwt_exp
from loguru import logger

def validate_token(func) -> None:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not check_jwt_exp(self.user_session.access_token):
            self.user_session.__reissue_token()
        return func(self, *args, **kwargs)

    return wrapper

class UserSession:
    def __init__(self, email: str, password: str):
        """Initialize the UserSession.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """

        self.email = email
        self.password = password
        self.__login()

    def __login(self) -> None:
        try:
            data = LoginRequest(username=self.email, password=self.password)
            response = self.client.login(data)
            self.access_token = response.access_token
            self.refresh_token = response.refresh_token
            logger.info("Login successful")

        except Exception as e:
            logger.error(f"Login failed. Error: {e}")
            raise e

    

    def __reissue_token(self) -> None:
        try:
            data = RefreshTokenRequest(access_token=self.access_token, refresh_token=self.refresh_token)
            response = self.client.refresh_token(data)
            self.access_token = response.access_token
            self.refresh_token = response.refresh_token

        except Exception as e:
            raise e


class BaseClient:
    def __init__(self, email: str, password: str):
        """Initialize with login information.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
        """
        self.user_session = UserSession(email=email, password=password)

    def __init__(self, user_session: UserSession):
        """Initialize with the User Session.

        Args:
            user_session (UserSession): 
        """
        self.user_session = user_session

    def validate_token(func) -> None:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not check_jwt_exp(self.user_session.access_token):
                self.user_session.__reissue_token()
            return func(self, *args, **kwargs)

        return wrapper

    @validate_token
    def get_credit(self) -> int:
        """Get the available NetsPresso credits.

        Raises:
            e: If an error occurs while getting credit information.

        Returns:
            int: The total amount of available NetsPresso credits.
        """

        try:
            credit = self.client.get_credit(access_token=self.access_token)
            logger.info(f"Get Credit successful. Credit: {credit.total}")

            return credit.total

        except Exception as e:
            logger.error(f"Get Credit failed. Error: {e}")
            raise e