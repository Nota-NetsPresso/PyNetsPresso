import sys


def check_credit_balance(user_credit, service_credit):
    service_name = service_credit.name.replace("_", " ").lower()
    if user_credit < service_credit:
        sys.exit(
            f"Your current balance of {user_credit} credits is insufficient to complete the task. \n{service_credit} credits are required for one {service_name} task. \nFor additional credit, please contact us at netspresso@nota.ai."
        )
