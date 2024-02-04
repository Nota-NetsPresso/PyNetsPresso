from pathlib import Path
from setuptools import setup, find_packages


version = (Path("netspresso") / "VERSION").read_text().strip()

long_description = Path("README.md").read_text(encoding="UTF8")

install_requires = Path("requirements.txt").read_text().split('\n')

setup(
    name="netspresso",
    version=version,
    author="NetsPresso",
    author_email="netspresso@nota.ai",
    description="PyNetsPresso",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nota-NetsPresso/PyNetsPresso",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",)),
    package_data={"netspresso.clients": ["configs/*.ini"], "netspresso": ["VERSION"]},
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
