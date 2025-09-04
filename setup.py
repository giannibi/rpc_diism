from setuptools import setup, find_packages

setup(
    name="rpc_diism",
    version="0.1.0",
    description="Template library for rpc_diism",
    author="",
    packages=find_packages(),
    install_requires=[
        "control",
        "slycot"
    ],
    python_requires=">=3.7",
)
