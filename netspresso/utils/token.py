from datetime import datetime
import jwt
import pytz

def check_jwt_exp(access_token):
    payload = jwt.decode(access_token, options={"verify_signature": False})
    return datetime.now(pytz.utc).timestamp() <= payload["exp"]
