import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tssearch",
    version="0.1.3",
    author="Fraunhofer Portugal",
    description="Library for time series subsequence search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/fraunhoferportugal/tssearch/archive/refs/tags/v0.1.3.tar.gz",
    package_data={'tssearch': ['distances/distances.json', 'examples/ecg.pickle']},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy', 'pandas', 'matplotlib', 'numpy'],
)
