from pathlib import Path

import setuptools

ROOT = Path(__file__).parent

with open("README.md") as fh:
    long_description = fh.read()


def find_requirements(filename):
    with (ROOT / "requirements" / filename).open() as f:
        return [s for s in [line.strip(" \n") for line in f] if not s.startswith("#") and s != ""]


install_requirements = find_requirements("requirements.txt")
docs_requirements = find_requirements("requirements-docs.txt")

setuptools.setup(
    name="tssearch",
    version="0.1.3",
    author="Fraunhofer Portugal",
    description="Library for time series subsequence search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/fraunhoferportugal/tssearch/archive/refs/tags/v0.1.3.tar.gz",
    package_data={"tssearch": ["distances/distances.json", "examples/ecg.pickle"]},
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=install_requirements,
    extras_require={
        "docs": docs_requirements,
    },
)
