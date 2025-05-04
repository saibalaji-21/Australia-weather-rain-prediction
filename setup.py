from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mlops",
    version="0.1",
    author="Sai",
    author_email="wihelmsai@gmail.com",
    description="MLOPS PROJECT",
    packages=find_packages(),
    install_requires=requirements,
)
