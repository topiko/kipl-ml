from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()


# Function to parse requirements.txt
def parse_requirements(filename: str):
    with open(filename, "r") as file:
        return [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]


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
    install_requires=parse_requirements(
        "requirements.txt"
    ),  # external packages as dependencies
)
