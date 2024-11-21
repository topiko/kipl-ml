from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="kipl-ml",
    version="0.0",
    description="Website fingerprinting and defences.",
    license="MIT",
    long_description=long_description,
    author="topiko",
    author_email="topiko1987@gmail.com",
    url="http://www.tbs/",
    packages=["kipl_ml"],  # same as name
    install_requires=[],  # external packages as dependencies
)
