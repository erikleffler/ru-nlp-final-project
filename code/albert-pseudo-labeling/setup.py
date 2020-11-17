from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "transformers>=3.4.0",
    "tensorboard_plugin_profile",
    "torch",
    "numpy",
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="trying out stuff with albert and BioASQ squad-like QA dataset",
)
