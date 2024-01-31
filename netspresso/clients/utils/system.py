import platform

from pkg_resources import DistributionNotFound, get_distribution


def get_package_version(package_name):
    try:
        version = get_distribution(package_name).version
        return version
    except DistributionNotFound:
        return None


def get_installed_packages_version(package_keys):
    packages_version = {}
    packages_version["python"] = platform.python_version()
    packages_version["os"] = f"{platform.system()} {platform.version()}"

    for package_key in package_keys:
        packages_version[package_key] = get_package_version(package_key)

    return packages_version


def generate_env_string(package_keys):
    packages_version = get_installed_packages_version(package_keys)
    env_str = "; ".join(f"{key} {value}" for key, value in packages_version.items() if value is not None)

    return env_str


PACKAGE_KEYS = ["torch", "tensorflow", "tensorflow-gpu", "numpy"]
ENV_STR = generate_env_string(PACKAGE_KEYS)
