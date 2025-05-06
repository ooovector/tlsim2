from setuptools import setup, find_packages

setup(
    name="tlsim2",
    version="0.0.1.dev1",
    url="https://github.com/ooovector/tlsim2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)
