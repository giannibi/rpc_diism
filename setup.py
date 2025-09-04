from setuptools import setup, find_packages

setup(
    name="rpc_diism",
    version="0.1.0",
    description="Python library for the Robust and Predictive Control course at DIISM, Unisi",
    author="Gianni Bianchini",
    packages=find_packages(),
    install_requires=[
        "control",
        "slycot"
    ],
    python_requires=">=3.10",
)
